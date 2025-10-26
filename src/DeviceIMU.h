#ifndef DeviceIMU_H
#define DeviceIMU_H
#include <Wire.h>

#include "FastIMU.h"

#define IMU_ADDRESS 0x68

class DeviceIMUClass {
 public:
	MPU9250 device;
	void setup(int sda, int scl);
};

extern DeviceIMUClass DeviceIMU;
#endif