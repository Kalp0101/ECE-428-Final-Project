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
