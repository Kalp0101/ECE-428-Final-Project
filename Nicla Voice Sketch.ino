#include "Nicla_System.h"
#include "NDP.h"

const int WAKE_TRIGGER_PIN = 5;

// Flag set by the callback, consumed by loop()
volatile bool wakeDetected = false;

void ledRedBlink() {
    while (1) {
        nicla::leds.begin();
        nicla::leds.setColor(red);
        delay(200);
        nicla::leds.setColor(off);
        delay(200);
        nicla::leds.end();
    }
}

// Keep this as short as possible — just set the flag and return immediately
void onKeywordMatch(String keyword) {
    wakeDetected = true;
}

void setup() {
    Serial.begin(115200);

    pinMode(WAKE_TRIGGER_PIN, OUTPUT);
    digitalWrite(WAKE_TRIGGER_PIN, LOW);

    nicla::begin();
    nicla::enable3V3LDO();

    NDP.onError(ledRedBlink);

    Serial.println("Loading Alexa synpackages...");
    NDP.begin("mcu_fw_120_v91.synpkg");
    NDP.load("dsp_firmware_v91.synpkg");
    NDP.load("alexa_334_NDP120_B0_v11_v91.synpkg");
    Serial.println("Packages loaded successfully.");

    NDP.turnOnMicrophone();
    NDP.onMatch(onKeywordMatch);
    NDP.interrupts();

    Serial.println("Listening for 'Alexa'...");
}

void loop() {
    NDP.poll();

    if (wakeDetected) {
        wakeDetected = false;

        Serial.println("Wake word detected!");

        // GPIO pulse to Raspberry Pi
        digitalWrite(WAKE_TRIGGER_PIN, HIGH);
        delay(500);
        digitalWrite(WAKE_TRIGGER_PIN, LOW);

        // Visual feedback — safe to use delay() here in loop()
        nicla::leds.begin();
        nicla::leds.setColor(green);
        delay(500);
        nicla::leds.setColor(off);
        nicla::leds.end();

        Serial.println("Cooldown complete. Listening again...");

        // Cooldown before re-arming to prevent double-triggers
        delay(1000);
    }
}
