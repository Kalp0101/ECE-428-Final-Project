# Final Project for Dr. Choi's ECE 428 Embedded Computer Systems
The goal of this project is to implement two actuators and two inputs in a complete Embedded System.

### Dr. Choi has a tough time remembering people's names. This project aims to address that by giving him a wearable device that performs the grueling task of remembering names and faces for him.

An onboard microphone detects inputs for the voice controls. A specific phrase can be spoken to begin scanning the person's face. Once sufficiently scanned, the speaker output plays a sound to indicate the completed scan and prompts the user to assign a name to the face, and saves the name and face to an onboard database. 

Next, the facial recognition system will be able to determine whether the person the camera is looking at is in the database or unknown. If they are saved in the database, the user can use voice commands to ask the device to read their name aloud to identify them.

### 2 Inputs:
microphone for voice commands, and pi camera.
- Look into voice-interfacing modules like the Arduino Nicla Voice -> [datasheet](https://docs.arduino.cc/hardware/nicla-voice/)
- [Nicla Voice ML code examples of using keyword triggers and built-in speech recognition](https://docs.arduino.cc/tutorials/nicla-voice/getting-started-ml/)

### 2 Outputs:
speaker for reading the name out loud, and a saved file of the person's face + name.
- [Video for implementing facial recognition on Raspberry Pi](https://www.youtube.com/watch?v=3TUlJrRJUeM)
- API calls to the phone for the language processing?

### Notes:
Another important aspect of this project is containerizing it using Docker. This GitHub repo acts as a quick place I can store my codebase and write down my ideas during the phase of the project. 

## Facial Recognition
### The Algorithm:
Use OpenCV's [Facial Recognition library](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html) using the Eigenfaces algorithm
- Definition: They are the eigenvectors of the covariance matrix of a set of face images, often visualized as ghost-like, grayscale images (eigenpictures) that highlight common variations, such as hair, eyes, and pose.
- Function: Instead of using every pixel to identify a face, the system projects face images into a lower-dimensional "face space" (subspace), creating a weighted combination of these eigenvectors.
- Recognition Process: A new face is recognized by comparing its specific weights (contributions of each eigenface) to those of known individuals in a database.
