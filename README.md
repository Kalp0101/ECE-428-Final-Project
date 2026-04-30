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

---

## Hardware

### Boards
- **Arduino Nicla Voice (ABX00061)** — always-on wake word detection via the onboard Syntiant NDP120 Neural Decision Processor. Draws ~0.8 mA in standby, powered by a dedicated 3.7V LiPo on the J4 connector.
- **Raspberry Pi 5** — runs all face recognition and speech transcription workloads. Powered by a USB power bank.

### Peripherals
- **Raspberry Pi Camera Module 3** — autofocus, 12MP IMX708 sensor. Used for face capture and encoding.
- **USB Microphone** — connected to the Pi for Whisper command transcription and name recording.
- **Bluetooth Speaker / Earbuds** — audio output for all spoken responses and name playback.

### Wiring

The Nicla Voice and Raspberry Pi 5 communicate over two wires only. No level shifter is required — both boards operate at 3.3V logic.

| Nicla Voice | Connector | Raspberry Pi 5 |
|---|---|---|
| LPIO0_EXT (P0_24, Arduino pin 5) | J1 Pin 1 | Physical Pin 11 (BCM GPIO 17) |
| GND | J2 Pin 6 | Physical Pin 9 (GND) |

When the wake word is detected, the Nicla pulses LPIO0_EXT HIGH for 500ms. The Pi detects this rising edge on BCM GPIO 17 and begins listening for a voice command.

> **Power note:** The Nicla runs from its own LiPo battery connected to J4 on the underside of the board. This allows it to listen for the wake word independently, even when the Pi is booting or idle.

---

## Repository Structure

```
.
├── Face_Scan_and_Recog.py          # Main Python application (Raspberry Pi)
├── Nicla_Voice_Sketch.ino          # Arduino sketch (Nicla Voice wake word trigger)
├── Autostart on Boot.ini           # systemd service file for boot autostart
├── Setup Commands After Service File.bash  # Shell commands for service setup
├── Dependencies and Packages.md   # Full list of required packages and install commands
└── README.md
```

---

## Code

### `Face_Scan_and_Recog.py` — Raspberry Pi Application

The main application running on the Pi. It idles in a low-CPU state, polling BCM GPIO 17 for a rising edge from the Nicla Voice. When triggered, it activates the Whisper speech recognition model to transcribe a voice command, then executes one of two workflows:

- **Enroll:** Captures a face embedding via the Pi Camera, records a 5-second audio clip of the person's name, and saves both to `faces.pkl` and `name_recordings/`.
- **Identify:** Captures a face embedding and compares it against the database using Euclidean distance. If a match is found (distance ≤ 0.6), it plays back the stored name audio clip.

Key design decisions:
- GPIO is handled via raw `lgpio` polling rather than `gpiozero` callbacks, which proved unreliable when running as a systemd service on the Pi 5.
- The face database uses atomic writes (`fsync` + `os.replace`) to prevent corruption on sudden power loss.
- The camera uses `Picamera2` instead of `cv2.VideoCapture` to properly support the Camera Module 3's ISP pipeline on the Pi 5.

### `Nicla_Voice_Sketch.ino` — Arduino Wake Word Trigger

Runs on the Arduino Nicla Voice. Loads the pre-trained Alexa keyword model onto the NDP120 and polls for detections. When "Alexa" is detected with sufficient confidence, it pulses Arduino pin 5 (LPIO0_EXT / P0_24) HIGH for 500ms, then enters a 1-second cooldown to prevent double-triggers.

```
NDP.begin("mcu_fw_120_v91.synpkg")
NDP.load("dsp_firmware_v91.synpkg")
NDP.load("alexa_334_NDP120_B0_v11_v91.synpkg")
```

The `.synpkg` files ship with the `Arduino_NiclaSenseVoice` library and are loaded from the board's onboard flash at runtime.

### `Autostart on Boot.ini` — systemd Service

A systemd unit file that starts `Face_Scan_and_Recog.py` automatically on every boot, without requiring a login session, display, or network connection. Key parameters:

- `ExecStartPre=/bin/sleep 5` — gives audio and GPIO subsystems time to initialize after boot.
- `Restart=on-failure` / `RestartSec=10` — automatically restarts after crashes with enough delay for lgpio to release the GPIO pin cleanly.
- `Environment=GPIOZERO_PIN_FACTORY=lgpio` — forces the correct pin factory on the Pi 5.

### `Setup Commands After Service File.bash` — Service Management

A shell script with the commands needed to install, enable, and manage the argus systemd service after writing the `.ini` file. Includes `daemon-reload`, `enable`, `start`, and `journalctl` commands for log monitoring.

---

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

---

## Voice Detection and Control
I implemented the wake word detection using a pre-trained "Alexa" model on the Nicla Voice's NDP120 processor. For the actual speech-to-text command transcription, I am using OpenAI's Faster-Whisper model "Tiny" (~39M parameters) running natively on the Pi in CPU / int8 mode.

### Supported voice commands:
| Command | Action |
|---|---|
| `"start scanning"` | Enroll a new face |
| `"who is this?"` | Identify the person in frame |
| `"what is their name?"` | Identify the person in frame |
| `"exit"` | Shut down the application |

---

## Libraries and Dependencies

### Raspberry Pi (Python)

| Library | Purpose |
|---|---|
| `face-recognition` | Face detection and 128-D embedding via dlib |
| `faster-whisper` | CPU-native speech-to-text (Whisper tiny.en, int8) |
| `picamera2` | Pi Camera Module 3 capture (libcamera-based) |
| `opencv-python` | Frame processing and colour conversion |
| `SpeechRecognition` | Microphone input and audio buffering |
| `lgpio` | Direct GPIO access on the Pi 5 |
| `sounddevice` | Name audio recording via the USB microphone |
| `scipy` | WAV file serialization for name recordings |
| `numpy` | Audio data conversion and face distance computation |

Install all dependencies:
```bash
pip install opencv-python face-recognition faster-whisper SpeechRecognition numpy lgpio sounddevice scipy --break-system-packages
sudo apt install -y ffmpeg python3-picamera2
```

### Arduino (Nicla Voice)

| Library / Package | Purpose |
|---|---|
| `Arduino_NiclaSenseVoice` | NDP120 firmware loading and keyword detection |
| `Nicla_System.h` | Nicla board initialization and LED control |
| `mcu_fw_120_v91.synpkg` | NDP120 MCU firmware (bundled with library) |
| `dsp_firmware_v91.synpkg` | NDP120 DSP firmware (bundled with library) |
| `alexa_334_NDP120_B0_v11_v91.synpkg` | Pre-trained Alexa keyword model |

Board package: **Arduino Mbed OS Nicla Boards** — install via Arduino IDE Board Manager.

---

# Flow Chart:
<img width="1093" height="669" alt="Embedded Final Project drawio" src="https://github.com/user-attachments/assets/efd781bb-5de2-4e31-90b1-6233b67db98b" />

---

# User Guide:
### Enroll New Face Steps
 
1. Say *"Alexa"* to the Nicla Voice
2. Hear the ready chime from the Pi
3. Say *"start scanning"*
4. Hold the person's face in front of the camera
5. The system detects the face automatically
6. Hear *"Say their name."*
7. Speak the person's name clearly into the microphone (5-second recording window)
8. Recording is saved. That person is now enrolled.

Repeat for each person. The `faces.pkl` database and `name_recordings/` folder are built incrementally. You can enroll multiple people in one session or across sessions.
 
### Identification Steps
 
1. Say *"Alexa"* to the Nicla Voice
2. Hear the ready chime
3. Say *"who is this?"* or *"what is their name?"*
4. Hold the person's face in front of the camera
5. Hear *"Their name is..."* followed by the recorded audio clip of their name
6. If not recognized, hear the not-recognized response

### Required Audio Files

The following `.m4a` files must be placed in the same directory as `Face_Scan_and_Recog.py`:

| File | Plays when... |
|---|---|
| `ready_chime.m4a` | Wake word is detected by the Nicla |
| `completed_scan.m4a` | Face scan begins |
| `analyzing_face.m4a` | Identification begins |
| `say_their_name.m4a` | Face captured — user should speak the name |
| `their_name_is.m4a` | Prefix before the recorded name playback |
| `shutting_down.m4a` | Exit command received |
| `unknown_command.m4a` | Command not recognized |
| `not_recognized.m4a` | Face not found in database |
| `camera_error.m4a` | Camera failed to open |
