#include <Arduino.h>
#define LOOP_DELAY 50	 // milliseconds

#include "PeripheralIMU.h"
#include "PeripheralWifi.h"

int log_metrics = 0;
char server_host[] = "10.112.145.138";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

int state_window[32] = {};
int windex = 0;
int frames_since_fall = 100;
WiFiClient client;

void setup() {
	Serial.begin(9600);
	delay(1000);
	Serial.println("Hello world");

	PeripheralIMU.setup(4, 5);	// SDA, SCL

	PeripheralWifi.connect(1);
	PeripheralWifi.waitForConnect(1);

	pinMode(LED_BUILTIN, OUTPUT);
	digitalWrite(LED_BUILTIN, HIGH);
	pinMode(0, OUTPUT);
	digitalWrite(0, HIGH);	// Wifi is on
}

void loop() {
	AccelData accelData;
	PeripheralIMU.device.update();
	PeripheralIMU.device.getAccel(&accelData);

	float acc_x = accelData.accelX;
	float acc_y = accelData.accelY;
	float acc_z = accelData.accelZ;
	float acc_mag = sqrtf(pow(acc_x, 2) + pow(acc_y, 2) + pow(acc_z, 2));

	int weightless = acc_mag < 0.6;
	int peak = acc_mag > 2.0;
	int stable = acc_mag > 0.7 && acc_mag < 1.3;

	if (weightless) {
		state_window[windex] = 1;
	} else if (peak) {
		state_window[windex] = 2;
	} else if (stable) {
		state_window[windex] = 3;
	} else {
		state_window[windex] = 0;
	}

	int fallen = 0;
	if (frames_since_fall < 50) {
		frames_since_fall++;
		fallen = 1;
	} else {
		for (int j = 0; j < 1; j++) {
			int windex_t = windex;
			int stables_remaining = 4;
			for (int i = 0; i < 4; i++) {
				if (state_window[windex_t] == 3)
					stables_remaining--;
				else
					stables_remaining = 4;
				windex_t = (windex_t - 1 + 32) % 32;
				if (stables_remaining <= 0) break;
			}
			if (stables_remaining > 0) break;

			int peaks_remaining = 1;
			for (int i = 0; i < 9 + 1; i++) {
				if (state_window[windex_t] == 2)
					peaks_remaining--;
				else
					peaks_remaining = 1;
				windex_t = (windex_t - 1 + 32) % 32;
				if (peaks_remaining <= 0) break;
			}
			if (peaks_remaining > 0) break;

			int weightless_remaining = 3;
			for (int i = 0; i < 5 + 3; i++) {
				if (state_window[windex_t] == 1)
					weightless_remaining--;
				else
					weightless_remaining = 3;
				windex_t = (windex_t - 1 + 32) % 32;
				if (weightless_remaining <= 0) break;
			}
			if (weightless_remaining > 0) break;
			fallen = 1;
			frames_since_fall = 0;
		}
	}

	if (frames_since_fall == 0) {
		if (client.connect(server_host, server_port)) {
			Serial.println(server_message);
			client.println(server_message);
			client.println();
			Serial.println("Sent fall message to server");
		} else {
			Serial.println("Connection to server failed");
		}
	}
	if (fallen == 1) {
		int odd = frames_since_fall & 1;
		digitalWrite(LED_BUILTIN, odd ? LOW : HIGH);
	}

	windex = (windex + 1) % 32;

	digitalWrite(0, PeripheralWifi.isConnected() ? HIGH : LOW);

	if (log_metrics) {
		Serial.print(">acc_mag:");
		Serial.println(acc_mag);

		Serial.print(">acc_x:");
		Serial.println(acc_x);
		Serial.print(">acc_y:");
		Serial.println(acc_y);
		Serial.print(">acc_z:");
		Serial.println(acc_z);

		Serial.println(">weightless:" + String(weightless));
		Serial.println(">peak:" + String(peak));
		Serial.println(">stable:" + String(stable));
		Serial.println(">fallen:" + String(fallen));
	}

	delay(LOOP_DELAY);
}