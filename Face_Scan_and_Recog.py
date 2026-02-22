import cv2
import face_recognition
import pickle
import os

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

def main():
    """
    Main execution loop. Continuously waits for user console input and 
    triggers the appropriate scanning or recognition workflows.
    """
    print("Facial Recognition System Started.")
    print("Commands are: 'Start scanning', 'Who is this?', 'What is their name?', or 'exit'")
    
    while True:
        # Continuously wait for user input from the console
        command = input("\nEnter command: ").strip()

        if command == "exit":
            print("Exiting system.")
            break

        elif command == "Start scanning":
            encoding = capture_single_face_encoding()
            if encoding is not None:
                # Ask the user for a name via console input
                name = input("Face captured successfully. What is their name? ").strip()
                if name:
                    # Save the face embedding and name into the persistent database
                    save_to_database(name, encoding)
                    print(f"Profile for '{name}' created and saved.")
                else:
                    print("Name cannot be empty. Capture discarded.")

        elif command in ["Who is this?", "What is their name?"]:
            encoding = capture_single_face_encoding()
            if encoding is not None:
                db = load_database()
                
                # Handle empty database case
                if not db["encodings"]:
                    print("I don't know")
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
                    
        else:
            print("Unknown command. Please enter an actual command.")

if __name__ == "__main__":
    main()
