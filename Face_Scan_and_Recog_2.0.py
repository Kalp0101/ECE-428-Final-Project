"""
ECE428 Final Project — Facial Recognition System
==================================================
Interrupt-driven facial recognition system designed to run on a Raspberry Pi.
Uses OpenCV for camera access and face_recognition for embedding generation/matching.
Stores known faces in a persistent pickle database (faces.pkl).
"""

import os
import sys
import signal
import pickle
import threading
import cv2
import face_recognition
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATABASE_FILE = "faces.pkl"          # Persistent storage for face embeddings
SCALE_FACTOR = 0.25                  # Resize to 1/4 for Pi performance
MATCH_TOLERANCE = 0.6                # face_recognition comparison threshold
CAMERA_INDEX = 0                     # Default webcam device index

# ─────────────────────────────────────────────
# Global state (for signal-handler access)
# ─────────────────────────────────────────────
_camera_lock = threading.Lock()      # Guards camera open/close
_active_camera = None                # Reference so signal handler can release


# ─────────────────────────────────────────────
# Signal / interrupt handling
# ─────────────────────────────────────────────
def _cleanup_handler(signum, frame):
    """
    Interrupt handler for SIGINT / SIGTERM.
    Releases the camera immediately and destroys any OpenCV windows
    so the device is never left in a locked state — critical for a
    battery-powered Raspberry Pi wearable where a pi camera drains power.
    """
    global _active_camera
    print("\n[INFO] Interrupt received — cleaning up...")
    with _camera_lock:
        if _active_camera is not None:
            _active_camera.release()
            _active_camera = None
    cv2.destroyAllWindows()
    sys.exit(0)


# Register for both SIGINT (Ctrl-C) and SIGTERM (kill / systemd stop)
signal.signal(signal.SIGINT, _cleanup_handler)
signal.signal(signal.SIGTERM, _cleanup_handler)


# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────
def load_database(path: str) -> dict:
    """
    Load the face database from a pickle file.
    Returns a dict with keys 'names' (list[str]) and 'encodings' (list[np.ndarray]).
    Creates an empty database if the file does not exist or is corrupted.
    """
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Validate structure
            if isinstance(data, dict) and "names" in data and "encodings" in data:
                print(f"[INFO] Loaded {len(data['names'])} face(s) from database.")
                return data
        except (pickle.UnpicklingError, EOFError, KeyError) as exc:
            print(f"[WARN] Database corrupted ({exc}); starting fresh.")
    # Return empty database if file missing or invalid
    return {"names": [], "encodings": []}


def save_database(path: str, data: dict) -> None:
    """Persist the face database to disk."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Database saved ({len(data['names'])} face(s)).")


# ─────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────
def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Open the webcam and store a global reference so the interrupt handler
    can release it if the user sends SIGINT mid-capture.
    """
    global _active_camera
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index {index}).")
    with _camera_lock:
        _active_camera = cap
    return cap


def close_camera(cap: cv2.VideoCapture) -> None:
    """Release camera and destroy preview windows."""
    global _active_camera
    with _camera_lock:
        if cap is not None:
            cap.release()
        _active_camera = None
    cv2.destroyAllWindows()


def capture_face(cap: cv2.VideoCapture) -> np.ndarray | None:
    """
    Continuously read frames from *cap*, display them, and attempt face detection.

    Processing pipeline per frame:
        1. Read BGR frame from camera.
        2. Resize to 1/4 resolution (SCALE_FACTOR) for faster detection on Pi.
        3. Convert BGR → RGB (face_recognition expects RGB).
        4. Detect face locations.
        5. If ≥1 face found → compute encoding for the *first* face and return it.

    The user can press 'q' in the preview window to abort.
    Returns the 128-d face encoding, or None if aborted / no face found.
    """
    print("[INFO] Camera active — looking for a face (press 'q' in window to cancel)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            return None

        # ── Show the live feed at full resolution ──
        cv2.imshow("Camera Feed", frame)

        # ── Down-scale for faster face detection ──
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        # ── Convert BGR (OpenCV default) → RGB (face_recognition expects) ──
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # ── Detect faces ──
        face_locations = face_recognition.face_locations(rgb_small)

        if face_locations:
            # Use only the first detected face
            print(f"[INFO] Detected {len(face_locations)} face(s) — using the first one.")
            encodings = face_recognition.face_encodings(rgb_small, face_locations)
            if encodings:
                return encodings[0]
            else:
                print("[WARN] Face detected but encoding failed; retrying...")

        # ── Allow window events & check for 'q' key to abort ──
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Capture cancelled by user.")
            return None


# ─────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────
def handle_start_scanning(db: dict) -> dict:
    """
    'Start scanning' workflow:
        1. Open camera → detect & encode a face.
        2. Prompt user for the person's name.
        3. Store the (name, encoding) pair in the database and persist to disk.
    Returns the (possibly updated) database dict.
    """
    try:
        cap = open_camera()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return db

    encoding = capture_face(cap)
    close_camera(cap)

    if encoding is None:
        print("[INFO] No face captured — nothing saved.")
        return db

    # ── Ask for a name (console input) ──
    name = input("Enter a name for this face: ").strip()
    if not name:
        print("[WARN] Empty name — discarding capture.")
        return db

    db["names"].append(name)
    db["encodings"].append(encoding)
    save_database(DATABASE_FILE, db)
    print(f"[INFO] Face for '{name}' registered successfully.")
    return db


def handle_identify(db: dict) -> None:
    """
    'Who is this?' / 'What is their name?' workflow:
        1. Check that the database is non-empty.
        2. Open camera → detect & encode a face.
        3. Compare against all stored encodings (tolerance = MATCH_TOLERANCE).
        4. Print the matched name or "I don't know".
    """
    if not db["encodings"]:
        print("[INFO] The database is empty — please scan a face first.")
        return

    try:
        cap = open_camera()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return

    encoding = capture_face(cap)
    close_camera(cap)

    if encoding is None:
        print("[INFO] No face captured — identification aborted.")
        return

    # ── Compare the captured encoding against every stored encoding ──
    matches = face_recognition.compare_faces(
        db["encodings"], encoding, tolerance=MATCH_TOLERANCE
    )

    if True in matches:
        # If multiple matches, pick the one with the smallest distance
        distances = face_recognition.face_distance(db["encodings"], encoding)
        best_idx = int(np.argmin(distances))
        if matches[best_idx]:
            print(f"Their name is {db['names'][best_idx]}")
        else:
            # Edge case: closest distance face wasn't in the boolean match list
            first_match_idx = matches.index(True)
            print(f"Their name is {db['names'][first_match_idx]}")
    else:
        print("I don't know")


# ─────────────────────────────────────────────
# Main event loop (interrupt-driven)
# ─────────────────────────────────────────────
def main() -> None:
    """
    Primary event loop.  Blocks on input() — which is effectively an
    interrupt-driven design: the program sleeps with zero CPU usage until
    the user (or a future hardware button via stdin pipe) provides a command.

    Supported commands:
        start scanning          — register a new face
        who is this?            — identify a face
        what is their name?     — identify a face (alias)
        exit                    — shut down gracefully
    """
    print("=" * 55)
    print("  Facial Recognition System  —  ECE428 Final Project")
    print("=" * 55)
    print("Commands:")
    print("  Start scanning        → Register a new face")
    print("  Who is this?          → Identify a face")
    print("  What is their name?   → Identify a face")
    print("  exit                  → Quit")
    print("=" * 55)

    # ── Load (or create) persistent database ──
    db = load_database(DATABASE_FILE)

    while True:
        try:
            user_input = input("\n> ").strip().lower()
        except EOFError:
            # stdin closed (e.g. pipe ended) — treat as exit
            break

        if user_input == "exit":
            print("[INFO] Exiting. Goodbye!")
            break

        elif user_input == "start scanning":
            db = handle_start_scanning(db)

        elif user_input in ("who is this?", "what is their name?"):
            handle_identify(db)

        elif user_input == "":
            continue  # ignore blank lines

        else:
            print("[WARN] Unknown command. Type 'Start scanning', "
                  "'Who is this?', 'What is their name?', or 'exit'.")


if __name__ == "__main__":
    main()
