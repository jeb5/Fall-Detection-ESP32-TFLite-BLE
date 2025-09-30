#include "PeripheralWifi.h"

PeripheralWifiClass::PeripheralWifiClass() {
};

void PeripheralWifiClass::connect(int verbose) {
	if (verbose) Serial.print("Connecting to WiFi...");
	WiFi.mode(WIFI_STA);
	struct station_config wifi_config;

	memset(&wifi_config, 0, sizeof(wifi_config));
	strcpy((char*)wifi_config.ssid, ssid);
	strcpy((char*)wifi_config.password, password);

	wifi_station_set_config(&wifi_config);
	wifi_set_macaddr(STATION_IF, target_esp_mac);
	wifi_station_set_wpa2_enterprise_auth(1);

	// Clean up to be sure no old data is still inside
	wifi_station_clear_cert_key();
	wifi_station_clear_enterprise_ca_cert();
	wifi_station_clear_enterprise_identity();
	wifi_station_clear_enterprise_username();
	wifi_station_clear_enterprise_password();
	wifi_station_clear_enterprise_new_password();

	wifi_station_set_enterprise_identity((uint8*)identity, strlen(identity));
	wifi_station_set_enterprise_username((uint8*)username, strlen(username));
	wifi_station_set_enterprise_password((uint8*)password, strlen(password));
	wifi_station_connect();
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