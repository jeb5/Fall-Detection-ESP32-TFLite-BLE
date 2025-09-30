#ifndef PeripheralIMU_H
#define PeripheralIMU_H
#include <Wire.h>

#include "FastIMU.h"

#define IMU_ADDRESS 0x68

class PeripheralIMUClass {
 public:
	MPU9250 device;
	void setup(int sda, int scl);
};

extern PeripheralIMUClass PeripheralIMU;
#endif