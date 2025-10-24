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
#define WINDOW_SIZE 64
// #define CIRCULARBUFFER_SIZE (WINDOW_SIZE * 2)
#define CIRCULARBUFFER_SIZE (163)
#define FALL_CONFIRMATION_COUNTDOWN 300
#define FALL_REPORTING_COUNTDOWN 800

bool log_metrics = false;
char server_host[] = "10.112.150.112";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

float circular_buffer[CIRCULARBUFFER_SIZE * 7] = {};
float supporting_circular_buffer[CIRCULARBUFFER_SIZE * 2] = {};
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
float dog_kernel[163];
long long loop_count = 0;

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

	int radius = 81;
	float sigma_small = 9.0f;
	float sigma_large = 27.0f;
	// x = np.arange(-radius, radius + 1)
	float sum = 0.0f;
	float small_gauss[163];
	for (int i = -radius; i <= radius; i++) {
		float value = exp(-0.5f * pow((i / sigma_small), 2));
		small_gauss[i + radius] = value;
		sum += value;
	}
	for (int i = 0; i < 2 * radius + 1; i++) {
		small_gauss[i] /= sum;
	}
	sum = 0.0f;
	float large_gauss[163];
	for (int i = -radius; i <= radius; i++) {
		float value = exp(-0.5f * pow((i / sigma_large), 2));
		large_gauss[i + radius] = value;
		sum += value;
	}
	for (int i = 0; i < 2 * radius + 1; i++) {
		large_gauss[i] /= sum;
	}
	for (int i = 0; i < 2 * radius + 1; i++) {
		dog_kernel[i] = small_gauss[i] - large_gauss[i];
	}

	for (int i = 0; i < CIRCULARBUFFER_SIZE; i++) {
		for (int j = 0; j < 7; j++) {
			circular_buffer[i * 7 + j] = 0.0f;
		}
		for (int j = 0; j < 2; j++) {
			supporting_circular_buffer[i * 2 + j] = 0.0f;
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

	int fi = (windex - 82 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
	{
		// We are interested in calculating the metrics in the middle of the window, at index 82 (fi)
		// We will calculate the acc low pass value at 82+15 = 97, using values from 82+30 = 112
		// We will calculate the angle change metric at 82, using values from 82+15 = 97
		float mag_acc = sqrtf(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
		supporting_circular_buffer[windex * 2 + 0] = mag_acc;

		static float acc_x_sum = 0.0f;
		static float acc_y_sum = 0.0f;
		static float acc_z_sum = 0.0f;
		static float angle_sum = 0.0f;

		int fiP15 = (fi + 15) % CIRCULARBUFFER_SIZE;
		int fiP30 = (fi + 30) % CIRCULARBUFFER_SIZE;
		int fiM15 = (fi - 15 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
		int fiM30 = (fi - 30 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;

		acc_x_sum += circular_buffer[fiP30 * 7 + 0];
		acc_x_sum -= circular_buffer[fi * 7 + 0];
		acc_y_sum += circular_buffer[fiP30 * 7 + 1];
		acc_y_sum -= circular_buffer[fi * 7 + 1];
		acc_z_sum += circular_buffer[fiP30 * 7 + 2];
		acc_z_sum -= circular_buffer[fi * 7 + 2];

		float acc_x_lp = acc_x_sum / 30.0f;
		float acc_y_lp = acc_y_sum / 30.0f;
		float acc_z_lp = acc_z_sum / 30.0f;

		float acc_x_P15 = circular_buffer[fiP15 * 7 + 0];
		float acc_y_P15 = circular_buffer[fiP15 * 7 + 1];
		float acc_z_P15 = circular_buffer[fiP15 * 7 + 2];

		float dot_prod = acc_z_P15 * acc_x_lp + acc_y_P15 * acc_y_lp + acc_z_P15 * acc_z_lp;

		float mag_acc_P15 = supporting_circular_buffer[fiP15 * 2 + 0];
		float mag_acc_lp = sqrtf(acc_x_lp * acc_x_lp + acc_y_lp * acc_y_lp + acc_z_lp * acc_z_lp);
		float cos_angle = dot_prod / (mag_acc_P15 * mag_acc_lp + 1e-6f);
		float angle = acosf(min(max(cos_angle, -1.0f), 1.0f));
		supporting_circular_buffer[fi * 1 + 0] = angle;
		angle_sum += angle;
		angle_sum -= supporting_circular_buffer[fiM30 * 1 + 0];
		float angle_lp = angle_sum / 30.0f;

		float dog_value = 0.0f;
		for (int i = 0; i < 163; i++) {
			int index = (windex - 82 + i + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
			float mag_acc_i = supporting_circular_buffer[index * 2 + 0];
			dog_value += mag_acc_i * dog_kernel[i];
		}
		float final_feature = dog_value * angle_lp;
		circular_buffer[fi * 7 + 6] = final_feature;

		if (final_feature > 0.07f) {
			Serial.println("High final feature value: " + String(final_feature, 6));
		}
	}

	// If the middle elements in the window has an acceleration magnitude > 2, there might be a fall.

	bool possible_fall = false;
	// float ax = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 1];
	// float ay = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 2];
	// float az = circular_buffer[(((windex - (WINDOW_SIZE / 2)) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE) * 7 + 3];
	// float acc_mag = sqrtf(pow(ax, 2) + pow(ay, 2) + pow(az, 2));
	// if (acc_mag > 2.0f) {
	// 	possible_fall = true;
	// 	Serial.println("Possible fall detected based on acceleration magnitude: " + String(acc_mag, 6));
	// }

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
		float acc_x_fi = circular_buffer[fi * 7 + 0];
		float acc_y_fi = circular_buffer[fi * 7 + 1];
		float acc_z_fi = circular_buffer[fi * 7 + 2];
		float custom_feature_fi = circular_buffer[fi * 7 + 6];

		Serial.println(">acc_x:" + String(acc_x_fi, 6));
		Serial.println(">acc_y:" + String(acc_y_fi, 6));
		Serial.println(">acc_z:" + String(acc_z_fi, 6));
		Serial.println(">custom_feature:" + String(custom_feature_fi, 6));
		Serial.println(">time_spent:" + String(time_spent));
		// Serial.println(">gyro_x:" + String(gyro_x, 6));
		// Serial.println(">gyro_y:" + String(gyro_y, 6));
		// Serial.println(">gyro_z:" + String(gyro_z, 6));
	}
	loop_count++;

	delay(time_to_next);
}