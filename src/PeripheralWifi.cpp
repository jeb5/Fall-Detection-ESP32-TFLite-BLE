#include "PeripheralWifi.h"

PeripheralWifiClass::PeripheralWifiClass() {
};

void PeripheralWifiClass::connect(int verbose) {
	if (verbose) Serial.print("Connecting to WiFi...");

	WiFi.mode(WIFI_STA);
	WiFi.begin(ssid, WPA2_AUTH_PEAP, identity, username, password);
}

void PeripheralWifiClass::waitForConnect(int verbose) {
	while (WiFi.status() != WL_CONNECTED) {
		delay(500);
		if (verbose) Serial.print(".");
	}
	if (verbose) {
		Serial.println();
		Serial.print("Connected, IP address: ");
		Serial.println(WiFi.localIP());
	}
}
int PeripheralWifiClass::isConnected() {
	return WiFi.status() == WL_CONNECTED;
}
PeripheralWifiClass PeripheralWifi;