# J.A.R.V.I.S.-Iron-Man-Helmet
Final Project for Dr. Choi's ECE 428 Embedded Computer Systems
The goal of this project is to implement two actuators and two inputs in a complete Embedded System.

I've chosen to build a J.A.R.V.I.S.-operated Iron Man helmet that implements speech recognition for the user interface and can control the opening and closing of the faceplate. It will also have an on-board camera that can perform facial recognition and identify people through a speaker.

2 Inputs: 
- The microphone for speech recognition
  - Look into voice-interfacing modules like the Arduino Nicla Voice -> [datasheet](https://docs.arduino.cc/hardware/nicla-voice/)
- The camera for facial recognition

2 Actuators:
- The motor that opens and closes the faceplate
- The Speaker that reads out the names of the people seen by the camera
   - [Video for implementing facial recognition on Raspberry Pi](https://www.youtube.com/watch?v=3TUlJrRJUeM)
   - Potentially replace the speaker with an arm-mounted turret that aims and shoots a laser at a target seen by the camera. Reference for that: [https://www.youtube.com/watch?v=a_UiYOO-Sd](https://youtu.be/a_UiYOO-Sdw?si=PaePHdlF61jSJcpm)
 
## UPDATE 1/27/2026 -> SHIFTING TO A NEW PROJECT
Dr.Choi has a tough time remembering people's names. This project aims to address that by giving him a wearable device that performs the grueling task of remembering names and faces for him.

An onboard microphone detects inputs for the voice controls. A specific phrase can be spoken to begin scanning the person's face. Once sufficiently scanned,the speaker output plays a sound to indicate the completed scan and prompts the user to assign a name to the face and saves the name and face to an onboard database. 

Next,the facial recognition will be able to detect whether the person the camera is looking at is someone from the database or unknown. If they are saved in the database,the user can use the voice commands to ask the device to read their name out loud to identify them.

2 Inputs:
voice commands and camera.

2 Outputs:
speaker when reading name and the saved file of the person's name.

- API calls to phone for the processing?
