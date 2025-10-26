#include "DeviceIMU.h"

calData calib = {1, {0, 0, 0.19}, {-0.31, 3.23, -1.00}, {-299.30, 190.60, -646.20}, {1.63, 0.76, 0.93}};
void DeviceIMUClass::setup(int sda, int scl) {
	Wire.begin(sda, scl);
	Wire.setClock(115200);

	int err = device.init(calib, IMU_ADDRESS);
	if (err != 0) {
		Serial.print("Error initializing IMU: ");
		Serial.println(err);
		while (true) {
			;
		}
	}
	Serial.println("Initialized IMU!");

	// SET THE ACCELEROMETER RANGE TO ±2g
	err = device.setAccelRange(2);
	if (err != 0) {
		Serial.print("Error Setting range: ");
		Serial.println(err);
		while (true) {
			;
		}
	}
}
DeviceIMUClass DeviceIMU;