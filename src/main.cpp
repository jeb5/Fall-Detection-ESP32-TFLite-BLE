#include <Arduino.h>
// #include <MadgwickAHRS.h>
#include <NimBLEDevice.h>

#include "PeripheralIMU.h"
#include "PeripheralWifi.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LOOP_DELAY 50		// milliseconds
#define GPIO_BUTTON 13	// GPIO pin for button input
#define WINDOW_SIZE 96
#define CIRCULARBUFFER_SIZE (WINDOW_SIZE * 2)
#define FALL_CONFIRMATION_COUNTDOWN 300
#define FALL_REPORTING_COUNTDOWN 800

bool log_metrics = false;
char server_host[] = "10.112.150.112";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

float circular_buffer[CIRCULARBUFFER_SIZE * 7] = {};
int windex = 0;
int frames_since_fall = FALL_REPORTING_COUNTDOWN + 1;
unsigned long previousMillis = 0;
WiFiClient client;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// Madgwick filter;
NimBLEAdvertising* pAdvertising;

constexpr int kTensorArenaSize = 3000 * 10;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
	Serial.begin(9600);
	delay(1000);
	Serial.println("Hello world");

	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	model = tflite::GetModel(model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		Serial.println("Model provided is schema version " + String(model->version()) + " not equal to supported version " + String(TFLITE_SCHEMA_VERSION) + ".");
		return;
	}
	static tflite::AllOpsResolver resolver;

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		Serial.println("AllocateTensors() failed");
		return;
	}

	input = interpreter->input(0);
	output = interpreter->output(0);

	PeripheralIMU.setup(27, 26);	// SDA, SCL

	// PeripheralWifi.connect(1);
	// PeripheralWifi.waitForConnect(1);

	NimBLEDevice::init("FallDetector");
	pAdvertising = NimBLEDevice::getAdvertising();
	pAdvertising->setName("FallDetector");
	pAdvertising->setMinInterval(100);
	pAdvertising->setMaxInterval(200);
	pAdvertising->setConnectableMode(BLE_GAP_CONN_MODE_NON);

	pinMode(GPIO_BUTTON, INPUT_PULLUP);	 // Button
	pinMode(25, OUTPUT);								 // BUZZER
	tone(25, 1000, 100);
	previousMillis = millis();
}

void loop() {
	frames_since_fall++;
	if (frames_since_fall < FALL_CONFIRMATION_COUNTDOWN) {
		// Frequency should increase as frames_since_fall increases to FALL_CONFIRMATION_COUNTDOWN
		int frequency = 800 + (800.0f * frames_since_fall) / FALL_CONFIRMATION_COUNTDOWN;
		if (frames_since_fall % 4 == 0) {
			tone(25, frequency, 90);
		}
	} else if (frames_since_fall == FALL_CONFIRMATION_COUNTDOWN) {
		tone(25, 3000, 500);
		// Fallen without cancelling
		Serial.println("Fall confirmed");
		if (PeripheralWifi.isConnected() and client.connect(server_host, server_port)) {
			Serial.println(server_message);
			client.println(server_message);
			client.println();
			Serial.println("Sent fall message to server");
		} else {
			Serial.println("Connection to server failed");
		}

		NimBLEAdvertisementData advertisementData;
		// "Fall=0"
		// char message[] = "Fall=1";
		// advertisementData.addData((uint8_t*)message, sizeof(message));
		// pAdvertising->setAdvertisementData(advertisementData);
		pAdvertising->start();
		Serial.println("Started advertising fall status");
	} else if (frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// Keep advertising fall status
		if (frames_since_fall % 12 == 0) {
			tone(25, 700, 500);
		}
	} else if (frames_since_fall > FALL_REPORTING_COUNTDOWN && pAdvertising->isAdvertising()) {
		pAdvertising->stop();
		Serial.println("Stopped advertising fall status");
	}

	unsigned long currentMillis = millis();
	unsigned long deltaTime = currentMillis - previousMillis;
	previousMillis = currentMillis;
	AccelData accelData;
	GyroData gyroData;
	PeripheralIMU.device.update();
	PeripheralIMU.device.getAccel(&accelData);
	PeripheralIMU.device.getGyro(&gyroData);

	float deltaTimeF = (float)deltaTime;
	float acc_x = accelData.accelX;
	float acc_y = accelData.accelY;
	float acc_z = accelData.accelZ;
	float gyro_x = gyroData.gyroX;
	float gyro_y = gyroData.gyroY;
	float gyro_z = gyroData.gyroZ;
	// filter.begin(1000.0f / deltaTimeF);
	// filter.updateIMU(gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z);
	// float roll = filter.getRoll();
	// float pitch = filter.getPitch();
	// float yaw = filter.getYaw();
	// Serial.println(">roll:" + String(roll, 6));
	// Serial.println(">pitch:" + String(pitch, 6));
	// Serial.println(">yaw:" + String(yaw, 6));
	// Serial.println(">deltaTime:" + String(deltaTimeF, 6));

	windex++;
	if (windex >= CIRCULARBUFFER_SIZE) {
		windex = 0;
	}

	int base_index = windex * 7;	// 7 features per timestep
	circular_buffer[base_index + 0] = deltaTimeF;
	circular_buffer[base_index + 1] = acc_x;
	circular_buffer[base_index + 2] = acc_y;
	circular_buffer[base_index + 3] = acc_z;
	circular_buffer[base_index + 4] = gyro_x;
	circular_buffer[base_index + 5] = gyro_y;
	circular_buffer[base_index + 6] = gyro_z;

	// If the middle elements in the window has an acceleration magnitude > 2, there might be a fall.

	bool possible_fall = false;
	float ax = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 1];
	float ay = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 2];
	float az = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 3];
	float acc_mag = sqrtf(pow(ax, 2) + pow(ay, 2) + pow(az, 2));
	if (acc_mag > 2.0f) {
		possible_fall = true;
		Serial.println("Possible fall detected based on acceleration magnitude: " + String(acc_mag, 6));
	}

	float inference_result = 0.0f;
	if (possible_fall && frames_since_fall > FALL_REPORTING_COUNTDOWN) {
		Serial.println("Running inference");
		unsigned long t_start = millis();
		// Need two copies because the circular buffer might wrap around
		int start_index = ((windex - WINDOW_SIZE + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7;
		int first_copy_size = min(CIRCULARBUFFER_SIZE * 7 - start_index, WINDOW_SIZE * 7);
		int second_copy_size = WINDOW_SIZE * 7 - first_copy_size;
		memcpy(input->data.f, circular_buffer + start_index, first_copy_size);
		if (second_copy_size > 0) {
			memcpy(input->data.f + first_copy_size, circular_buffer, second_copy_size);
		}

		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			Serial.println("Invoke failed! oh no :(");
		} else {
			unsigned long t_end = millis();
			inference_result = output->data.f[0];
			Serial.println("Inference result: " + String(inference_result, 6) + " (took " + String(t_end - t_start) + " ms)");
		}
	}

	if (inference_result > 0.4f) {
		Serial.println("Fall detected!");
		frames_since_fall = 0;
	}

	int button_value = digitalRead(GPIO_BUTTON) == HIGH ? 0 : 1;
	if (button_value == 1 && frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// If button is pressed, and we are in fall countdown, cancel the fall (though it may be too late if already reported)
		frames_since_fall = FALL_REPORTING_COUNTDOWN + 1;
		Serial.println("Fall cancelled by button press");
		tone(25, 500, 800);
	}

	unsigned long time_spent = millis() - currentMillis;
	unsigned long time_to_next = LOOP_DELAY > time_spent ? LOOP_DELAY - time_spent : 0;

	if (log_metrics) {
		Serial.println(">acc_x:" + String(acc_x, 6));
		Serial.println(">acc_y:" + String(acc_y, 6));
		Serial.println(">acc_z:" + String(acc_z, 6));
		Serial.println(">gyro_x:" + String(gyro_x, 6));
		Serial.println(">gyro_y:" + String(gyro_y, 6));
		Serial.println(">gyro_z:" + String(gyro_z, 6));
		Serial.println(">inference_result:" + String(inference_result, 6));
		Serial.print(">loop_time:");
		Serial.println(time_spent);
	}

	delay(time_to_next);
}