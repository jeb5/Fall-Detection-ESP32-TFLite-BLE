#ifndef PeripheralWifi_H
#define PeripheralWifi_H
// #include <ESP8266WiFi.h>
#include <WiFi.h>

// extern "C" {
// #include "c_types.h"
// #include "user_interface.h"
// #include "wpa2_enterprise.h"
// }

class PeripheralWifiClass {
 public:
	PeripheralWifiClass();
	void connect(int verbose);
	void waitForConnect(int verbose);
	int isConnected();
};

extern PeripheralWifiClass PeripheralWifi;
extern char ssid[];
extern char identity[];
extern char username[];
extern char password[];

#endif