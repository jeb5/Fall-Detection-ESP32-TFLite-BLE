#ifndef DeviceWifi_H
#define DeviceWifi_H
#include <WiFi.h>

class DeviceWifiClass {
 public:
	DeviceWifiClass();
	void connect(int verbose);
	void waitForConnect(int verbose);
	int isConnected();
};

extern DeviceWifiClass DeviceWifi;
extern char ssid[];
extern char identity[];
extern char username[];
extern char password[];

#endif