#include <Arduino.h>
#include <MadgwickAHRS.h>

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

int log_metrics = 0;
char server_host[] = "10.112.150.112";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

float data_window1[WINDOW_SIZE * 7] = {};
float data_window2[WINDOW_SIZE * 7] = {};
int windex1 = 0;
int windex2 = WINDOW_SIZE / 2;
int frames_since_fall = 100;
unsigned long previousMillis = 0;
WiFiClient client;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
Madgwick filter;

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

	PeripheralWifi.connect(1);
	PeripheralWifi.waitForConnect(1);

	pinMode(GPIO_BUTTON, INPUT_PULLUP);	 // Button
	pinMode(25, OUTPUT);								 // BUZZER
	tone(25, 1000, 100);
	previousMillis = millis();
}

void loop() {
	if (frames_since_fall < 100 && frames_since_fall % 3 == 0) {
		tone(25, 1000, 60);
	}
	frames_since_fall++;
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
	filter.begin(1000.0f / deltaTimeF);
	filter.updateIMU(gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z);
	float roll = filter.getRoll();
	float pitch = filter.getPitch();
	float yaw = filter.getYaw();
	Serial.println(">roll:" + String(roll, 6));
	Serial.println(">pitch:" + String(pitch, 6));
	Serial.println(">yaw:" + String(yaw, 6));
	Serial.println(">acc_x:" + String(acc_x, 6));
	Serial.println(">acc_y:" + String(acc_y, 6));
	Serial.println(">acc_z:" + String(acc_z, 6));
	Serial.println(">gyro_x:" + String(gyro_x, 6));
	Serial.println(">gyro_y:" + String(gyro_y, 6));
	Serial.println(">gyro_z:" + String(gyro_z, 6));
	// Serial.println(">deltaTime:" + String(deltaTimeF, 6));

	windex1 = (windex1 + 1) % WINDOW_SIZE;
	windex2 = (windex1 + (WINDOW_SIZE / 2)) % WINDOW_SIZE;

	float* windows[] = {data_window1, data_window2};
	int windexes[] = {windex1, windex2};
	for (int w = 0; w < 2; w++) {
		float* data_window = windows[w];
		int windex = windexes[w];
		int base_index = windex * 7;	// 7 features per timestep
		data_window[base_index + 0] = deltaTimeF;
		data_window[base_index + 1] = acc_x;
		data_window[base_index + 2] = acc_y;
		data_window[base_index + 3] = acc_z;
		data_window[base_index + 4] = gyro_x;
		data_window[base_index + 5] = gyro_y;
		data_window[base_index + 6] = gyro_z;
	}

	bool window1_full = windex1 == WINDOW_SIZE - 1;
	bool window2_full = windex2 == WINDOW_SIZE - 1;

	bool fallen = false;
	if (window1_full || window2_full) {
		// if (window1_full) {
		float* data_window = window1_full ? data_window1 : data_window2;
		Serial.println("Running inference on " + String(window1_full ? "window 1" : "window 2"));
		unsigned long t_start = millis();
		memcpy(input->data.f, data_window, sizeof(float) * WINDOW_SIZE * 7);

		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			Serial.println("Invoke failed! oh no :(");
		} else {
			unsigned long t_end = millis();
			float y = output->data.f[0];
			// Serial.println("Result: " + String(y, 6) + " (inference time: " + String(t_end - t_start) + " ms)");
			Serial.println(">inference_result:" + String(y, 6));

			if (PeripheralWifi.isConnected() && client.connect(server_host, server_port)) {
				client.println("POST /log_data HTTP/1.0");
				client.println("Content-Type: application/octet-stream");
				client.println("Content-Length: " + String(sizeof(float) * WINDOW_SIZE * 7));
				client.println();
				client.write((uint8_t*)data_window, sizeof(float) * WINDOW_SIZE * 7);
				client.println();
				Serial.println("Sent data window to server");
			} else {
				Serial.println("Connection to server failed");
			}

			if (frames_since_fall > 200) {	// avoid repeated fall detections
				fallen = y > 0.5;
			}
		}
	}

	if (fallen) {
		Serial.println("Fall detected!");
		frames_since_fall = 0;
		// if (PeripheralWifi.isConnected() and client.connect(server_host, server_port)) {
		// 	Serial.println(server_message);
		// 	client.println(server_message);
		// 	client.println();
		// 	Serial.println("Sent fall message to server");
		// } else {
		// 	Serial.println("Connection to server failed");
		// }
	}

	// digitalWrite(0, PeripheralWifi.isConnected() ? HIGH : LOW);

	int button_value = digitalRead(GPIO_BUTTON) == HIGH ? 0 : 1;

	unsigned long time_spent = millis() - currentMillis;
	Serial.print(">loop_time:");
	Serial.println(time_spent);
	unsigned long time_to_next = LOOP_DELAY > time_spent ? LOOP_DELAY - time_spent : 0;

	delay(time_to_next);
}