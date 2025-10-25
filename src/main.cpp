#include <Arduino.h>
// #include <MadgwickAHRS.h>
#include <NimBLEDevice.h>

#include <vector>

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

#define INFERENCE_WINDOW_SIZE 64
#define DOG_RADIUS 60
#define DOG_SIZE (DOG_RADIUS * 2 + 1)
#define CIRCULARBUFFER_SIZE 192
#define FALL_FEATURE_THRESHOLD 0.07f

#define FALL_CONFIRMATION_COUNTDOWN 300
#define FALL_REPORTING_COUNTDOWN 800

#define LOGGING 1

#if LOGGING
#define LOG(x) Serial.println(x)
#else
#define LOG(x)
#endif

char server_host[] = "10.112.150.112";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

float circular_buffer[CIRCULARBUFFER_SIZE * 7] = {};
float supporting_circular_buffer[CIRCULARBUFFER_SIZE * 4] = {};
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
float dog_kernel[DOG_SIZE];
long long loop_count = 0;

constexpr int kTensorArenaSize = 3000 * 10;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
	if (LOGGING == 1) {
		Serial.begin(9600);
		delay(1000);
		LOG("Hello world");
	}

	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	model = tflite::GetModel(model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		LOG("Model provided is schema version " + String(model->version()) + " not equal to supported version " + String(TFLITE_SCHEMA_VERSION) + ".");
		return;
	}
	static tflite::AllOpsResolver resolver;

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		LOG("AllocateTensors() failed");
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

	float sigma_small = 6.6f;
	float sigma_large = sigma_small * 3;
	float sum = 0.0f;
	float small_gauss[121];
	for (int i = -DOG_RADIUS; i <= DOG_RADIUS; i++) {
		float value = exp(-0.5f * pow((i / sigma_small), 2));
		small_gauss[i + DOG_RADIUS] = value;
		sum += value;
	}
	for (int i = 0; i < 2 * DOG_RADIUS + 1; i++) {
		small_gauss[i] /= sum;
	}
	sum = 0.0f;
	float large_gauss[121];
	for (int i = -DOG_RADIUS; i <= DOG_RADIUS; i++) {
		float value = exp(-0.5f * pow((i / sigma_large), 2));
		large_gauss[i + DOG_RADIUS] = value;
		sum += value;
	}
	for (int i = 0; i < 2 * DOG_RADIUS + 1; i++) {
		large_gauss[i] /= sum;
	}
	for (int i = 0; i < 2 * DOG_RADIUS + 1; i++) {
		dog_kernel[i] = small_gauss[i] - large_gauss[i];
	}

	for (int i = 0; i < CIRCULARBUFFER_SIZE; i++) {
		for (int j = 0; j < 7; j++) {
			circular_buffer[i * 7 + j] = 0.0f;
		}
		for (int j = 0; j < 2; j++) {
			supporting_circular_buffer[i * 4 + j] = 0.0f;
		}
	}

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
		LOG("Fall confirmed");
		if (PeripheralWifi.isConnected() and client.connect(server_host, server_port)) {
			LOG(server_message);
			client.println(server_message);
			client.println();
			LOG("Sent fall message to server");
		} else {
			LOG("Connection to server failed");
		}

		NimBLEAdvertisementData advertisementData;
		pAdvertising->start();
		LOG("Started advertising fall status");
	} else if (frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// Keep advertising fall status
		if (frames_since_fall % 12 == 0) {
			tone(25, 700, 500);
		}
	} else if (frames_since_fall > FALL_REPORTING_COUNTDOWN && pAdvertising->isAdvertising()) {
		pAdvertising->stop();
		LOG("Stopped advertising fall status");
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

	windex++;
	if (windex >= CIRCULARBUFFER_SIZE) {
		windex = 0;
	}

	int base_index = windex * 7;	// 7 features per timestep
	circular_buffer[base_index + 0] = acc_x;
	circular_buffer[base_index + 1] = acc_y;
	circular_buffer[base_index + 2] = acc_z;
	circular_buffer[base_index + 3] = gyro_x;
	circular_buffer[base_index + 4] = gyro_y;
	circular_buffer[base_index + 5] = gyro_z;
	circular_buffer[base_index + 6] = 0;

	float inference_result = 0.0f;
	static std::vector<long long> fall_indices;

	int fi = (windex - DOG_RADIUS + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
	{
		// We are interested in calculating the metrics in the middle of the window, at index 61 (fi)
		// We will calculate the acc low pass value at fi+8, using values from fi+8-19=fi-11 to fi+8+20=fi+28
		// We will calculate the angle change metric at fi, using values from fi-7 to fi+8
		float mag_acc = sqrtf(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
		supporting_circular_buffer[windex * 4 + 0] = mag_acc;

		static float acc_x_sum = 0.0f;
		static float acc_y_sum = 0.0f;
		static float acc_z_sum = 0.0f;
		static float angle_sum = 0.0f;

		int fiP8 = (fi + 8) % CIRCULARBUFFER_SIZE;
		int fiM12 = (fi - 12 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
		int fiP28 = (fi + 28) % CIRCULARBUFFER_SIZE;
		int fiM8 = (fi - 8 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;

		acc_x_sum += circular_buffer[fiP28 * 7 + 0];
		acc_x_sum -= circular_buffer[fiM12 * 7 + 0];
		acc_y_sum += circular_buffer[fiP28 * 7 + 1];
		acc_y_sum -= circular_buffer[fiM12 * 7 + 1];
		acc_z_sum += circular_buffer[fiP28 * 7 + 2];
		acc_z_sum -= circular_buffer[fiM12 * 7 + 2];

		float acc_x_lp = acc_x_sum / 40.0f;
		float acc_y_lp = acc_y_sum / 40.0f;
		float acc_z_lp = acc_z_sum / 40.0f;

		float acc_xP8 = circular_buffer[fiP8 * 7 + 0];
		float acc_yP8 = circular_buffer[fiP8 * 7 + 1];
		float acc_zP8 = circular_buffer[fiP8 * 7 + 2];

		float dot_prod = acc_xP8 * acc_x_lp + acc_yP8 * acc_y_lp + acc_zP8 * acc_z_lp;

		float mag_acc_P8 = supporting_circular_buffer[fiP8 * 4 + 0];
		float mag_acc_lp = sqrtf(acc_x_lp * acc_x_lp + acc_y_lp * acc_y_lp + acc_z_lp * acc_z_lp);
		float cos_angle = dot_prod / (mag_acc_P8 * mag_acc_lp + 1e-6f);
		float angle = acosf(min(max(cos_angle, -1.0f), 1.0f));
		supporting_circular_buffer[fiP8 * 4 + 1] = angle;
		angle_sum += angle;
		angle_sum -= supporting_circular_buffer[fiM8 * 4 + 1];
		float angle_lp = angle_sum / 16.0f;
		supporting_circular_buffer[fi * 4 + 2] = angle_lp;

		float dog_value = 0.0f;
		for (int i = 0; i < DOG_SIZE; i++) {
			int index = (fi - DOG_RADIUS + i + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
			float mag_acc_i = supporting_circular_buffer[index * 4 + 0];
			dog_value += mag_acc_i * dog_kernel[i];
		}
		supporting_circular_buffer[fi * 4 + 3] = dog_value;
		float final_feature = dog_value * angle_lp;
		circular_buffer[fi * 7 + 6] = final_feature;

		static float max_feature_current_fall = -1.0f;
		static long long max_feature_index = 0;

		if (final_feature > FALL_FEATURE_THRESHOLD && loop_count > CIRCULARBUFFER_SIZE) {	 // Falling
			if (final_feature > max_feature_current_fall) {
				max_feature_current_fall = final_feature;
				max_feature_index = loop_count - DOG_RADIUS;
			}
		} else if (max_feature_index != 0 && (final_feature < FALL_FEATURE_THRESHOLD)) {	// Stopped falling
			// Record the fall event
			fall_indices.push_back(max_feature_index);
			max_feature_current_fall = -1.0f;
			max_feature_index = 0;
		}
	}

	// If the middle elements in the window has an acceleration magnitude > 2, there might be a fall.

	bool possible_fall = false;
	// float ax = circular_buffer[(((windex - (INFERENCE_WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 1];
	// float ay = circular_buffer[(((windex - (INFERENCE_WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 2];
	// float az = circular_buffer[(((windex - (INFERENCE_WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 3];
	// float acc_mag = sqrtf(pow(ax, 2) + pow(ay, 2) + pow(az, 2));
	// if (acc_mag > 2.0f) {
	// 	possible_fall = true;
	// 	LOG("Possible fall detected based on acceleration magnitude: " + String(acc_mag, 6));
	// }

	// float inference_result = 0.0f;
	// if (possible_fall && frames_since_fall > FALL_REPORTING_COUNTDOWN) {
	// 	LOG("Running inference");
	// 	unsigned long t_start = millis();
	// 	// Need two copies because the circular buffer might wrap around
	// 	int start_index = ((windex - INFERENCE_WINDOW_SIZE + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7;
	// 	int first_copy_size = min(CIRCULARBUFFER_SIZE * 7 - start_index, INFERENCE_WINDOW_SIZE * 7);
	// 	int second_copy_size = INFERENCE_WINDOW_SIZE * 7 - first_copy_size;
	// 	memcpy(input->data.f, circular_buffer + start_index, first_copy_size);
	// 	if (second_copy_size > 0) {
	// 		memcpy(input->data.f + first_copy_size, circular_buffer, second_copy_size);
	// 	}

	// 	TfLiteStatus invoke_status = interpreter->Invoke();
	// 	if (invoke_status != kTfLiteOk) {
	// 		LOG("Invoke failed! oh no :(");
	// 	} else {
	// 		unsigned long t_end = millis();
	// 		inference_result = output->data.f[0];
	// 		LOG("Inference result: " + String(inference_result, 6) + " (took " + String(t_end - t_start) + " ms)");
	// 	}
	// }

	if (inference_result > 0.4f) {
		LOG("Fall detected!");
		frames_since_fall = 0;
	}

	int button_value = digitalRead(GPIO_BUTTON) == HIGH ? 0 : 1;
	if (button_value == 1 && frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// If button is pressed, and we are in fall countdown, cancel the fall (though it may be too late if already reported)
		frames_since_fall = FALL_REPORTING_COUNTDOWN + 1;
		LOG("Fall cancelled by button press");
		tone(25, 500, 800);
	}

	if (true) {
		fi = (windex - 160 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
		int fi_li = loop_count - 160;
		int fall = 0;

		if (!fall_indices.empty()) {
			int first_fall_index = fall_indices.front();
			if (fi_li >= first_fall_index) {
				fall_indices.erase(fall_indices.begin());
			}
			if (fi_li == first_fall_index) {
				fall = 1;
			}
		}

		float acc_x_fi = circular_buffer[fi * 7 + 0];
		float acc_y_fi = circular_buffer[fi * 7 + 1];
		float acc_z_fi = circular_buffer[fi * 7 + 2];
		float gyro_x_fi = circular_buffer[fi * 7 + 3];
		float gyro_y_fi = circular_buffer[fi * 7 + 4];
		float gyro_z_fi = circular_buffer[fi * 7 + 5];
		float custom_feature_fi = circular_buffer[fi * 7 + 6];
		float acc_mag_fi = supporting_circular_buffer[fi * 4 + 0];
		float angle_lp_fi = supporting_circular_buffer[fi * 4 + 2];
		float dog_value_fi = supporting_circular_buffer[fi * 4 + 3];
		float angle_fi = supporting_circular_buffer[fi * 4 + 1];

		// LOG(">acc_x:" + String(acc_x_fi, 6));
		// LOG(">acc_y:" + String(acc_y_fi, 6));
		// LOG(">acc_z:" + String(acc_z_fi, 6));
		// LOG(">custom_feature:" + String(custom_feature_fi, 6));
		// LOG(">time_spent:" + String(time_spent));
		// LOG(">gyro_x:" + String(gyro_x, 6));
		// LOG(">gyro_y:" + String(gyro_y, 6));
		// LOG(">gyro_z:" + String(gyro_z, 6));

		// LOG(String(currentMillis) + "," + String(button_value) + "," + String(acc_x_fi, 6) + "," + String(acc_y_fi, 6) + "," + String(acc_z_fi, 6) + "," + String(gyro_x_fi, 6) + "," + String(gyro_y_fi, 6) + "," + String(gyro_z_fi, 6) + "," + String(custom_feature_fi, 6) + "," + String(acc_mag_fi, 6) + "," + String(angle_lp_fi, 6) + "," + String(dog_value_fi, 6) + "," + String(angle_fi, 6) + "," + String(fall));
	}
	unsigned long time_spent = millis() - currentMillis;
	LOG("time_spent:" + String(time_spent) + " ms");
	unsigned long time_to_next = LOOP_DELAY > time_spent ? LOOP_DELAY - time_spent : 0;
	loop_count++;

	delay(time_to_next);
}