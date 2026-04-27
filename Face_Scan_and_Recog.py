#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import os
import sys
import subprocess
import speech_recognition as sr
from faster_whisper import WhisperModel
import numpy as np
from gpiozero import Button
import threading
import argparse
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
import warnings
from contextlib import contextmanager
from typing import Optional

# ── Terminal Silence & Error Suppression ──────────────────────────────────────
# Suppress pkg_resources deprecation warnings only — scoped to avoid hiding
# meaningful warnings from face_recognition or faster-whisper.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

@contextmanager
def suppress_system_errors():
    """
    Temporarily redirects low-level C library errors (like JACK, ALSA, and Qt)
    to /dev/null to keep the terminal clean.

    This works at the file descriptor level so it catches noise from C libraries
    that bypass Python's warning system entirely. It is safe here because
    _processing_lock ensures on_wake_triggered() only runs on one thread at a
    time — no concurrent thread will lose an error to the redirect.
    """
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stderr_fd = os.dup(sys.stderr.fileno())
    try:
        os.dup2(null_fd, sys.stderr.fileno())
        yield
    finally:
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        os.close(null_fd)
        os.close(old_stderr_fd)


# ── Deployment Flag ───────────────────────────────────────────────────────────
# Set to True before final deployment in the enclosed, headless unit.
# Disables all cv2.imshow() calls that require a display server.
HEADLESS = False

# ── Registration Flag (set via argparse in main()) ────────────────────────────
# Controls whether face enrollment is permitted. Set True only when running
# with --register flag (i.e., during setup with a microphone available).
REGISTRATION_ENABLED = False

# ── GPIO Configuration ────────────────────────────────────────────────────────
# BCM GPIO 17 = Physical Pin 11 on the Pi 5.
# Receives a 200ms HIGH pulse from the Nicla Voice on wake word detection.
# pull_up=False: Nicla drives the pin actively. bounce_time debounces the pulse.
WAKE_GPIO_PIN = 17
wake_button = Button(WAKE_GPIO_PIN, pull_up=False, bounce_time=0.3)

# ── Audio Recording Configuration ────────────────────────────────────────────
SAMPLE_RATE = 16000        # Hz — matches Whisper's expected input rate
NAME_RECORDING_SECONDS = 5 # Duration of the name recording prompt
RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "name_recordings"
)

# ── Audio Response Files ──────────────────────────────────────────────────────
# All .m4a files must live in the same directory as this script.
#
# Required files:
#   ready_chime.m4a      — plays when Nicla wake trigger fires
#   completed_scan.m4a   — plays when face scan begins
#   analyzing_face.m4a   — plays when identification begins
#   say_their_name.m4a   — prompts user to record the person's name
#   their_name_is.m4a    — "Their name is..." prefix before name playback
#   shutting_down.m4a    — plays on exit command
#   unknown_command.m4a  — plays on unrecognised command
#   not_recognized.m4a   — plays when face is not in the database
#   camera_error.m4a     — plays if the camera fails to open
AUDIO_RESPONSES = {
    "wake":             "ready_chime.m4a",
    "start scanning":   "completed_scan.m4a",
    "start scan":       "completed_scan.m4a",
    "who is this":      "analyzing_face.m4a",
    "say their name":   "say_their_name.m4a",
    "their name is":    "their_name_is.m4a",
    "exit":             "shutting_down.m4a",
    "unknown command":  "unknown_command.m4a",
    "not recognized":   "not_recognized.m4a",
    "camera error":     "camera_error.m4a",
}

# ── Script Directory Helper ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(file_path: str) -> str:
    """
    Resolve a file path to an absolute path.
    If file_path is already absolute, return it unchanged.
    Otherwise, join it with the script's directory.
    """
    if os.path.isabs(file_path):
        return file_path
    return os.path.join(SCRIPT_DIR, file_path)


# ── Audio Playback ────────────────────────────────────────────────────────────

def play_audio(file_path: str):
    """
    Play an audio file asynchronously (non-blocking).
    Use for one-shot cues where the program should continue immediately.
    """
    resolved = _resolve_path(file_path)
    if os.path.exists(resolved):
        try:
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", resolved]
            )
        except Exception as e:
            print(f"Error playing audio: {e}")
    else:
        print(f"Warning: Audio file not found at {resolved}")


def play_audio_sync(file_path: str):
    """
    Play an audio file synchronously (blocking).
    Use when the next action must wait until playback finishes — e.g., playing
    "Their name is..." immediately before the recorded name clip.
    """
    resolved = _resolve_path(file_path)
    if os.path.exists(resolved):
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", resolved],
                check=True
            )
        except Exception as e:
            print(f"Error playing audio: {e}")
    else:
        print(f"Warning: Audio file not found at {resolved}")


# ── Whisper Model ─────────────────────────────────────────────────────────────
print("Loading Whisper model...")
try:
    with suppress_system_errors():
        whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("Whisper model loaded.")
except Exception as e:
    print(f"Warning: Could not load Whisper model. {e}")
    whisper_model = None


# ── Face Database ─────────────────────────────────────────────────────────────
# Database schema:
#   {
#       "encodings":   [np.ndarray, ...],  # 128-d face embeddings
#       "name_audio":  [str, ...],         # absolute paths to .wav name recordings
#   }
#
# "name_audio" replaces the old "names" text field. Each entry is the absolute
# path to a .wav file containing a recording of that person's name.

DB_FILE = os.path.join(SCRIPT_DIR, "faces.pkl")


def load_database() -> dict:
    """
    Load the face database from disk.
    Returns an empty database if the file does not exist or is empty.
    Handles legacy databases that used a 'names' text field by migrating
    them to the new 'name_audio' schema with a placeholder path.
    """
    if not os.path.exists(DB_FILE):
        return {"encodings": [], "name_audio": []}

    with open(DB_FILE, "rb") as f:
        try:
            db = pickle.load(f)
        except EOFError:
            return {"encodings": [], "name_audio": []}

    # ── Legacy migration ──────────────────────────────────────────────────────
    # If the database was saved with the old 'names' text field, convert it.
    # Old entries won't have audio files, so we mark them as needing re-enrollment.
    if "names" in db and "name_audio" not in db:
        print("Notice: Migrating legacy database (text names → audio paths).")
        db["name_audio"] = [
            f"[legacy: {n} — re-enroll to record audio]"
            for n in db["names"]
        ]
        del db["names"]
        save_database(db)

    return db


def save_database(db: dict):
    """
    Persist the database dict to disk using an atomic write.

    Write to a temp file first, fsync to force the OS to flush to the SD card,
    then atomically replace the real file. This ensures that a power cut mid-write
    never corrupts faces.pkl — the old file stays intact until the new one is
    fully written.
    """
    temp_file = DB_FILE + ".tmp"

    with open(temp_file, "wb") as f:
        pickle.dump(db, f)
        f.flush()             # Flush Python's internal write buffer
        os.fsync(f.fileno())  # Force the OS to physically write to the SD card

    os.replace(temp_file, DB_FILE)  # Atomic swap — no partial-write window


def add_to_database(encoding: np.ndarray, name_audio_path: str):
    """Append a new face encoding and its name audio path to the database."""
    db = load_database()
    db["encodings"].append(encoding)
    db["name_audio"].append(name_audio_path)
    save_database(db)
    print(f"Profile saved. Database now contains {len(db['encodings'])} face(s).")


# ── Name Audio Recording ──────────────────────────────────────────────────────

def record_name_audio() -> Optional[str]:
    """
    Record NAME_RECORDING_SECONDS of audio from the default microphone and save
    it as a .wav file in the RECORDINGS_DIR folder.

    Flow:
        1. Play "say_their_name.m4a" synchronously so the user hears the prompt
           before the recording window opens.
        2. Record for NAME_RECORDING_SECONDS seconds.
        3. Save and return the absolute file path, or None on failure.
    """
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # Timestamp-based filename guarantees uniqueness across enrollments
    filename = f"name_{int(time.time())}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    # Block until the prompt finishes — recording starts immediately after
    play_audio_sync(AUDIO_RESPONSES["say their name"])
    print(f"Recording name ({NAME_RECORDING_SECONDS}s)... speak now.")

    try:
        with suppress_system_errors():
            recording = sd.rec(
                frames=int(NAME_RECORDING_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()  # Block until the full recording is captured

        wavfile.write(filepath, SAMPLE_RATE, recording)
        print(f"Name recording saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error recording name audio: {e}")
        return None


# ── Face Capture ──────────────────────────────────────────────────────────────

def capture_single_face_encoding(timeout_seconds: int = 10) -> Optional[np.ndarray]:
    """
    Activate the camera and wait until a face is detected, up to timeout_seconds.
    Returns the 128-d embedding of the first detected face, or None on failure.

    The timeout prevents an infinite hang in headless mode when no face is present.
    Display window and keyboard abort are suppressed when HEADLESS is True.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        play_audio(AUDIO_RESPONSES["camera error"])
        return None

    print(f"Camera activated. Waiting up to {timeout_seconds}s for a face...")
    face_encoding = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout_seconds:
            print("[WARN] Camera timeout: No face detected.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        if not HEADLESS:
            with suppress_system_errors():
                cv2.imshow("Camera Feed - Waiting for Face", frame)

        # Downscale to 1/4 size for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if len(face_locations) > 0:
            print("Face detected! Extracting embedding...")
            encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if len(encodings) > 0:
                face_encoding = encodings[0]
                break

        # Allow manual abort via 'q' key when a display is present
        if not HEADLESS:
            with suppress_system_errors():
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Camera capture manually aborted.")
                    break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    return face_encoding


# ── Voice Command Listener ────────────────────────────────────────────────────

def listen_for_command() -> str:
    """
    Listen via microphone and transcribe with Whisper (tiny.en, CPU, int8).
    Returns the lowercased, punctuation-stripped transcription.
    Falls back to console input if the Whisper model failed to load.
    """
    if whisper_model is None:
        return input("\nEnter command (Whisper unavailable — keyboard fallback): ").strip().lower()

    recognizer = sr.Recognizer()
    try:
        with suppress_system_errors():
            with sr.Microphone(sample_rate=16000) as source:
                print("Listening for command...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    except Exception as e:
        print(f"Microphone error: {e}")
        return ""

    print("Processing voice...")
    try:
        audio_data = (
            np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = whisper_model.transcribe(audio_data, beam_size=1, language="en")
        transcription = " ".join([seg.text for seg in segments]).strip().lower()
        for char in [".", ",", "?", "!"]:
            transcription = transcription.replace(char, "")
        print(f"Heard: '{transcription}'")
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


# ── Command Handler ───────────────────────────────────────────────────────────

def handle_command(command: str) -> bool:
    """
    Process a single transcribed voice command.
    Returns False if the system should shut down, True to keep running.
    """
    if not command:
        return True

    # ── Exit ──────────────────────────────────────────────────────────────────
    if "exit" in command:
        print("Exiting system.")
        play_audio_sync(AUDIO_RESPONSES["exit"])
        return False

    # ── Face Registration ─────────────────────────────────────────────────────
    elif "start scanning" in command or "start scan" in command:

        if not REGISTRATION_ENABLED:
            # Enrollment is intentionally disabled in deployed/headless mode.
            # The database should be fully populated before deployment.
            print("Registration is disabled in recognition-only mode.")
            play_audio(AUDIO_RESPONSES["unknown command"])
            return True

        play_audio(AUDIO_RESPONSES["start scanning"])
        encoding = capture_single_face_encoding()

        if encoding is not None:
            name_audio_path = record_name_audio()
            if name_audio_path is not None:
                add_to_database(encoding, name_audio_path)
                print("Face and name recording saved to database.")
            else:
                print("Name recording failed. Face capture discarded.")
        else:
            print("No face captured. Enrollment cancelled.")

    # ── Face Identification ───────────────────────────────────────────────────
    elif (
        "who is this" in command
        or "what is their name" in command
        or "what's their name" in command
    ):
        play_audio(AUDIO_RESPONSES["who is this"])
        encoding = capture_single_face_encoding()

        if encoding is not None:
            db = load_database()

            if not db["encodings"]:
                print("Database is empty — no faces enrolled yet.")
                play_audio(AUDIO_RESPONSES["not recognized"])
                return True

            face_distances = face_recognition.face_distance(db["encodings"], encoding)
            best_match_index = int(np.argmin(face_distances))
            best_distance = face_distances[best_match_index]

            if best_distance <= 0.6:
                name_audio_path = db["name_audio"][best_match_index]
                print(f"Match found (distance: {best_distance:.3f}). Playing name.")

                # Play "Their name is..." then the recorded name clip back-to-back.
                # Both calls are synchronous so the clips play sequentially with no gap.
                play_audio_sync(AUDIO_RESPONSES["their name is"])
                play_audio_sync(name_audio_path)
            else:
                print(f"No match found (closest distance: {best_distance:.3f}).")
                play_audio(AUDIO_RESPONSES["not recognized"])

    # ── Unknown ───────────────────────────────────────────────────────────────
    else:
        print(f"Unknown command: '{command}'")
        play_audio(AUDIO_RESPONSES["unknown command"])

    return True


# ── Wake Trigger Callback ─────────────────────────────────────────────────────

_running = True
_processing_lock = threading.Lock()


def on_wake_triggered():
    """
    Called by gpiozero on a rising edge on GPIO 17 (Nicla Voice wake pulse).
    Runs automatically in a background thread managed by gpiozero.

    Uses a threading.Lock() (non-blocking acquire) to ignore any new wake pulses
    that arrive while a command is already being processed. The lock is always
    released in the finally block, even if an exception occurs mid-command.

    Sequence:
        1. Play ready chime (async — audible cue that the system is active)
        2. Short pause to let the chime finish before opening the microphone
        3. Listen for a voice command via Whisper
        4. Handle the command
    """
    global _running

    # Try to acquire the lock without blocking. If another trigger is already
    # being processed, acquire() returns False immediately and we return early.
    if not _processing_lock.acquire(blocking=False):
        print("[WAKE] Ignored — system is already processing a command.")
        return

    try:
        print("\n[WAKE] Nicla Voice triggered. Ready for command.")
        play_audio(AUDIO_RESPONSES["wake"])

        # Wait for the ready chime to finish before activating the microphone.
        # Adjust if your ready_chime.m4a is longer or shorter than ~0.8s.
        time.sleep(0.8)

        command = listen_for_command()
        should_continue = handle_command(command)

        if not should_continue:
            _running = False
            wake_button.when_pressed = None  # Detach handler to stop further triggers

    finally:
        # Always release the lock — even if an exception occurred mid-command.
        _processing_lock.release()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    global REGISTRATION_ENABLED, _running

    parser = argparse.ArgumentParser(
        description="Facial Recognition System"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "Enable face registration mode. "
            "Use this during setup (with microphone available) to enroll faces. "
            "Do not pass this flag on the deployed, enclosed unit."
        ),
    )
    args = parser.parse_args()
    REGISTRATION_ENABLED = args.register

    print("=" * 50)
    print("  Facial Recognition System")
    print("=" * 50)

    if REGISTRATION_ENABLED:
        print("  Mode: REGISTRATION (face enrollment enabled)")
    else:
        print("  Mode: RECOGNITION ONLY (enrollment disabled)")

    if HEADLESS:
        print("  Display: HEADLESS (camera preview suppressed)")
    else:
        print("  Display: ACTIVE (camera preview enabled)")

    print(f"  Wake GPIO: BCM {WAKE_GPIO_PIN}")
    print(f"  Name recordings: {RECORDINGS_DIR}")
    print("=" * 50)
    print(f"\nWaiting for wake word trigger on GPIO {WAKE_GPIO_PIN}...")
    print("(Say your wake word to the Nicla Voice to activate)\n")

    # Attach the rising-edge callback
    wake_button.when_pressed = on_wake_triggered

    # Keep the main thread alive until the exit command is received.
    # All work happens inside on_wake_triggered() via gpiozero's background thread.
    while _running:
        time.sleep(0.1)

    print("\nSystem shut down cleanly.")


if __name__ == "__main__":
    main()
