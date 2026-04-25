### Install these dependencies onto the Raspberry Pi:

~~~
sudo apt update
sudo apt install -y ffmpeg python3-pip python3-lgpio
~~~
### And these Python packages:
```
pip install \
    opencv-python \
    face-recognition \
    faster-whisper \
    SpeechRecognition \
    numpy \
    gpiozero \
    lgpio \
    sounddevice \
    scipy \
    --break-system-packages
```
