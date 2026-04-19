// ─────────────────────────────────────────────────────────────────────────────
// Argus — Nicla Voice Wake Word Trigger
//
// Listens for the wake word "awaken" using the NDP120 Neural Decision Processor.
// When detected, pulses LPIO0_EXT (J1 Pin 1) HIGH for 200ms to trigger the Pi.
// ─────────────────────────────────────────────────────────────────────────────

#include <Arduino_NiclaSenseVoice.h>
// Replace this with the exact header filename from your Edge Impulse export.
// Open the downloaded .zip, look inside src/, and use that filename here.
#include <awaken_detector_inferencing.h>

// LPIO0_EXT = J1 Pin 1 on the Nicla Voice.
// Referenced as A6 in the Arduino Mbed board package (MKR A6 compatibility pin).
const int WAKE_TRIGGER_PIN = A6;

NiclaSenseVoice voice;
volatile bool wakeWordDetected = false;

// Called by the NiclaSenseVoice library when inference produces a result.
// Runs in interrupt context — keep it short.
void onWakeWordDetected(const char* label, float score) {
    // "awaken" must exactly match the class label used in Edge Impulse.
    // It is case-sensitive and usually lowercase with underscores for spaces.
    if (strcmp(label, "awaken") == 0 && score > 0.85f) {
        wakeWordDetected = true;
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);  // Allow serial monitor to connect

    pinMode(WAKE_TRIGGER_PIN, OUTPUT);
    digitalWrite(WAKE_TRIGGER_PIN, LOW);

    Serial.println("Initializing Nicla Voice...");

    if (!voice.begin()) {
        Serial.println("ERROR: NiclaSenseVoice initialization failed.");
        while (true) { delay(1000); }
    }

    voice.setWakeWordCallback(onWakeWordDetected);
    Serial.println("Listening for wake word: 'awaken'");
}

void loop() {
    // voice.update() feeds audio to the NDP120 and runs inference.
    // Do NOT add delay() here — it will starve the audio pipeline.
    voice.update();

    if (wakeWordDetected) {
        wakeWordDetected = false;

        Serial.println("Wake word detected! Pulsing trigger pin.");

        // Pulse HIGH for 200ms — long enough for gpiozero's interrupt to catch.
        digitalWrite(WAKE_TRIGGER_PIN, HIGH);
        delay(200);
        digitalWrite(WAKE_TRIGGER_PIN, LOW);

        // Cooldown: prevents double-triggering from a single utterance.
        delay(1000);

        Serial.println("Listening again...");
    }
}
