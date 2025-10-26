#include "DeviceWifi.h"

DeviceWifiClass::DeviceWifiClass() {
};

void DeviceWifiClass::connect(int verbose) {
	if (verbose) Serial.print("Connecting to WiFi...");

	WiFi.mode(WIFI_STA);
	WiFi.begin(ssid, WPA2_AUTH_PEAP, identity, username, password);
}

void DeviceWifiClass::waitForConnect(int verbose) {
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
int DeviceWifiClass::isConnected() {
	return WiFi.status() == WL_CONNECTED;
}
DeviceWifiClass DeviceWifi;