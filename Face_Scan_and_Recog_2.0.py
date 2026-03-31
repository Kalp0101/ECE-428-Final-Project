import cv2
import face_recognition
import pickle
import os
import subprocess
import speech_recognition as sr
from faster_whisper import WhisperModel
import numpy as np

# Dictionary mapping detected phrases to their corresponding audio files
AUDIO_RESPONSES = {
    "start scanning": "completed_scan.m4a",
    "start scan": "completed_scan.m4a",
    "who is this": "analyzing_face.m4a",
    "what is their name": "type_out_their_name.m4a",
    "exit": "shutting_down.m4a",
    "unknown command": "unknown_command.m4a",
    "not recognized": "not_recognized.m4a",
    "camera error": "camera_error.m4a"
}

def play_audio(file_name):
    """
    Plays an M4A audio file through the default system audio (Bluetooth).
    Relies on ffmpeg being installed on the Raspberry Pi.
    """
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if os.path.exists(audio_path):
        try:
            # Using ffplay from ffmpeg to play the file asynchronously without a display
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path])
        except Exception as e:
            print(f"Error playing audio: {e}")
    else:
        print(f"Warning: Audio file not found at {audio_path}")

# Initialize Whisper model (optimized for Raspberry Pi - CPU, int8, tiny model)
print("Loading Whisper model...")
try:
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
except Exception as e:
    print(f"Warning: Could not load whisper model. {e}")
    whisper_model = None

# File to store the facial embeddings database
# Use an absolute path so the database is always found regardless of a working directory
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.pkl")

def load_database():
    """
    Loads the facial embeddings database from a pickle file.
    If the file does not exist or is empty, returns an empty database structure.
    """
    if not os.path.exists(DB_FILE):
        return {"names": [], "encodings": []}
    
    with open(DB_FILE, "rb") as f:
        try:
            return pickle.load(f)
        except EOFError:
            # Empty file case
            return {"names": [], "encodings": []}

def save_to_database(name, encoding):
    """
    Saves a new name and its corresponding facial embedding to the pickle file.
    """
    db = load_database()
    db["names"].append(name)
    db["encodings"].append(encoding)
    
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def capture_single_face_encoding():
    """
    Activates the camera, continuously displays the feed, and waits until 
    at least one face is detected. Captures the first detected face, creates
    an embedding, and returns it. Closes the camera when done.
    """
    cap = cv2.VideoCapture(0)
    
    # Handle camera error stuff
    if not cap.isOpened():
        print("Error: Could not access the camera. Lock in dude.")
        play_audio(AUDIO_RESPONSES["camera error"])
        return None

    print("Camera activated. Waiting for a face...")
    face_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from cam.")
            break

        # Continuously display the camera feed
        cv2.imshow("Camera Feed - Waiting for Face", frame)
        
        # Resize frame to 1/4 size for processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert OpenCV's BGR color format to RGB for face_recognition
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Wait until at least one face is detected
        if len(face_locations) > 0:
            print("Face detected! Extracting embedding...")
            # Generate facial embeddings for the detected faces
            encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            if len(encodings) > 0:
                # Capture the first detected face only
                face_encoding = encodings[0]
                break

        # OpenCV requires waitKey to update the display window so allow the user to manually abort with 'q' if stuck
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Camera capture manually aborted.")
            break

    # Clean up and close the camera
    cap.release()
    cv2.destroyAllWindows()
    
    return face_encoding

def listen_for_command():
    """
    Listens to the microphone and uses faster_whisper to transcribe voice to text.
    Returns the lowercased transcription. Optimized for Raspberry Pi.
    """
    if whisper_model is None:
        return input("\nEnter command (fallback due to model error): ").strip().lower()
        
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("\nListening for voice command...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen to the user with a timeout to prevent hanging
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    except Exception as e:
        print(f"Microphone error: {e}")
        return ""

    print("Processing voice...")
    try:
        # Get raw 16-bit PCM audio data and convert to 32-bit float for Whisper
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with a tiny beam size for faster execution on Pi
        segments, _ = whisper_model.transcribe(audio_data, beam_size=1, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip().lower()
        
        # Strip simple punctuation to make matching easier
        for char in [".", ",", "?", "!"]:
            transcription = transcription.replace(char, "")
            
        print(f"Heard: '{transcription}'")
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def main():
    """
    Main execution loop. Continuously waits for user console input and 
    triggers the appropriate scanning or recognition workflows.
    """
    print("Facial Recognition System Started.")
    print("Commands are: 'Start scanning', 'Who is this?', 'What is their name?', or 'exit'")
    
    while True:
        # Continuously listen for user audio input from the console instead of typing
        command = listen_for_command()
        
        # In case nothing was recorded or model failed
        if not command:
            continue

        if "exit" in command:
            print("Exiting system.")
            play_audio(AUDIO_RESPONSES["exit"])
            break

        elif "start scanning" in command or "start scan" in command:
            play_audio(AUDIO_RESPONSES["start scanning"])
            encoding = capture_single_face_encoding()
            if encoding is not None:
                # Ask the user for a name via terminal input to prevent typos with voice models
                play_audio(AUDIO_RESPONSES["what is their name"])
                name = input("Face captured successfully. What is their name? ").strip()
                if name:
                    # Save the face embedding and name into the persistent database
                    save_to_database(name, encoding)
                    print(f"Profile for '{name}' created and saved.")
                else:
                    print("Name cannot be empty. Capture discarded.")

        elif "who is this" in command or "what is their name" in command or "what's their name" in command:
            play_audio(AUDIO_RESPONSES["who is this"])
            encoding = capture_single_face_encoding()
            if encoding is not None:
                db = load_database()
                
                # Handle empty database case
                if not db["encodings"]:
                    print("I don't know")
                    play_audio(AUDIO_RESPONSES["not recognized"])
                    continue
                
                # Use face distances to find the best match
                import numpy as np
                face_distances = face_recognition.face_distance(db["encodings"], encoding)
                best_match_index = int(np.argmin(face_distances))
                best_distance = face_distances[best_match_index]

                # Tolerance of 0.6 for now
                if best_distance <= 0.6:
                    matched_name = db["names"][best_match_index]
                    print(f"Their name is {matched_name} (distance: {best_distance:.3f})")
                else:
                    print(f"I don't know (closest distance: {best_distance:.3f})")
                    play_audio(AUDIO_RESPONSES["not recognized"])
                    
        else:
            print("Unknown command. Please enter an actual command.")
            play_audio(AUDIO_RESPONSES["unknown command"])

if __name__ == "__main__":
    main()
