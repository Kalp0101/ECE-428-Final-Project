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
Another important aspect of this project is containerizing it using Docker. This GitHub repo acts as a quick place I can store my codebase and write down my ideas during the phase of the project. Also, I am documenting the codebase as I research the algorithms I plan to use.

## Facial Recognition
### The Algorithm:
Uses the **face_recognition** [library](https://github.com/ageitgey/face_recognition), which is a high-level wrapper around dlib, which itself uses a deep convolutional neural network (CNN). The pipeline has three distinct stages:

#### **Stage 1:** Face Detection

By default, this uses dlib's Histogram of Oriented Gradients (HOG), which is a widely used feature descriptor, combined with a linear Support Vector Machine (SVM) for object detection

HOG: The image is divided into small cells. Within each cell, the gradient direction and magnitude of every pixel is computed. These gradients are binned into a histogram of orientations (typically 9 bins). The result is a compact descriptor of local texture/shape.

Linear SVM: The HOG descriptor is fed into an SVM trained to classify regions as "face" or "not face". A sliding window scans the image at multiple scales to detect faces of varying sizes.
We resize to 1/4 since face_locations is O(n^2) with respect to image dimensions. Shrinking the frame 4x reduces computation by ~16x at the cost of detecting only larger faces.
Using a max-pooling CNN is significantly more accurate but much slower.

#### **Stage 2:** Face Encoding

This is a two-step process and is the core of the algorithm.

1. Face Alignment via Landmark Detection

Before encoding, dlib runs a 68-point facial landmark predictor (a regression tree ensemble) to identify key points: corners of eyes, tip of nose, jawline, etc. The face is then affine-transformed, rotated, scaled, and cropped so that eyes and mouth are always in the same canonical positions. This normalization is critical; without it, the same face at different angles would produce very different embeddings.

2. Deep Metric Learning with a ResNet

The aligned face is passed through a ResNet-34-like CNN (29 convolutional layers) that was trained using metric learning (specifically a variant of triplet loss):

$$
\mathcal{L} = \max \left(0,\ \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha \right)
$$

Where:
A = anchor face (a person)
P = positive sample (same person, different photo)
N = negative sample (different person)
α = margin (enforces separation)
The network is trained to minimize distance between embeddings of the same person and maximize distance between embeddings of different people. After training on millions of faces, the network outputs a 128-dimensional unit vector, which is a point on a hypersphere in R^128 that uniquely represents a face.

#### **Stage 3:** Recognition via Euclidean Distance (Eigenfaces)

$$
d(e_1, e_2) = \|e_1 - e_2\|_2
= \sqrt{\sum_{i=1}^{128} (e_{1i} - e_{2i})^2}
$$

Since all embeddings are unit-normalized, this Euclidean distance is directly related to **cosine similarity**. The threshold `0.6` means that if two embeddings are within a distance of 0.6 in the 128-D space, they are considered the same person. Empirically:

<div align="center">

| Distance | Interpretation |
|----------|---------------|
| < 0.4    | Very confident match |
| 0.4 – 0.6 | Likely match |
| > 0.6    | Likely different people |

</div>

### To summarize:
Raw frame → HOG/SVM detection → 68-point landmark alignment → ResNet-34 embedding → 128-D vector → Euclidean distance comparison

## Voice Detection and Control
I'm gonna try to implement it using a custom keyword detection model on the Nicla Voice, but if I can't finish it in time, I'll rely on [Nvidia's Nemotron Speech-to-Text model (600m)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) or [OpenAI's Whisper Small English (244m)](https://huggingface.co/openai/whisper-small.en) running on the Pi. 

# User Guide:
### Enrollment Steps (per person)
 
1. Say *"Alexa"* to the Nicla Voice
2. Hear the ready chime from the Pi
3. Say *"start scanning"*
4. Hold the person's face in front of the camera
5. The system detects the face automatically — no button press needed
6. Hear *"Say their name."*
7. Speak the person's name clearly into the microphone (5-second recording window)
8. Recording is saved. That person is now enrolled.
Repeat for each person. The `faces.pkl` database and `name_recordings/` folder are built incrementally — you can enroll multiple people in one session or across sessions.
 
### Identification (deployed mode)
 
1. Say *"Alexa"* to the Nicla Voice
2. Hear the ready chime
3. Say *"who is this?"* or *"what is their name?"*
4. Hold the person's face in front of the camera
5. Hear *"Their name is..."* followed by the recorded audio clip of their name
6. If not recognized, hear the not-recognized response
