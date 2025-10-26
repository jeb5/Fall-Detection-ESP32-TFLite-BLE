#define DATA_COLLECTION_MODE 0
#if !DATA_COLLECTION_MODE
#include <Arduino.h>
#include <NimBLEDevice.h>

#include <vector>

#include "DeviceIMU.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LOOP_DELAY 50		// milliseconds
#define GPIO_BUTTON 13	// GPIO pin for button input
#define GPIO_BUZZER 25	// GPIO pin for buzzer output

#define INFERENCE_WINDOW_SIZE 64
#define DOG_RADIUS 60
#define DOG_SIZE (DOG_RADIUS * 2 + 1)
#define CIRCULARBUFFER_SIZE 192
#define FALL_FEATURE_THRESHOLD 0.07f
#define CBC 4
#define SBC 2

#define FALL_CONFIRMATION_COUNTDOWN 300
#define FALL_REPORTING_COUNTDOWN 800

#define LOGGING 0

#if LOGGING
#define LOG(x) Serial.println(x)
#else
#define LOG(x)
#endif

float circular_buffer[CIRCULARBUFFER_SIZE * CBC] = {};
float supporting_circular_buffer[CIRCULARBUFFER_SIZE * SBC] = {};
int windex = 0;
long long frames_since_fall = FALL_REPORTING_COUNTDOWN + 1;
unsigned long previousMillis = 0;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// Madgwick filter;
NimBLEAdvertising* pAdvertising;
float dog_kernel[DOG_SIZE];
long long loop_count = 0;

#define kTensorArenaSize 20000
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
#if LOGGING
	Serial.begin(9600);
	delay(1000);
	LOG("Hello world");
#endif

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

	DeviceIMU.setup(27, 26);	// SDA, SCL

	NimBLEDevice::init("FallDetector");
	pAdvertising = NimBLEDevice::getAdvertising();
	pAdvertising->setName("FallDetector");
	pAdvertising->setMinInterval(100);
	pAdvertising->setMaxInterval(200);
	pAdvertising->setConnectableMode(BLE_GAP_CONN_MODE_NON);

	// Set up DoG kernel
	float sigma_small = 6.6f;
	float sigma_large = sigma_small * 3;
	float sum = 0.0f;
	float small_gauss[DOG_SIZE];
	for (int i = -DOG_RADIUS; i <= DOG_RADIUS; i++) {
		float value = exp(-0.5f * pow((i / sigma_small), 2));
		small_gauss[i + DOG_RADIUS] = value;
		sum += value;
	}
	for (int i = 0; i < DOG_SIZE; i++) {
		small_gauss[i] /= sum;
	}
	sum = 0.0f;
	float large_gauss[DOG_SIZE];
	for (int i = -DOG_RADIUS; i <= DOG_RADIUS; i++) {
		float value = exp(-0.5f * pow((i / sigma_large), 2));
		large_gauss[i + DOG_RADIUS] = value;
		sum += value;
	}
	for (int i = 0; i < DOG_SIZE; i++) {
		large_gauss[i] /= sum;
	}
	for (int i = 0; i < DOG_SIZE; i++) {
		dog_kernel[i] = small_gauss[i] - large_gauss[i];
	}

	pinMode(GPIO_BUTTON, INPUT_PULLUP);	 // Button
	pinMode(GPIO_BUZZER, OUTPUT);				 // BUZZER
	tone(GPIO_BUZZER, 1000, 100);
	previousMillis = millis();
}

void loop() {
	frames_since_fall++;
	if (frames_since_fall < FALL_CONFIRMATION_COUNTDOWN) {
		// Frequency should increase as frames_since_fall increases to FALL_CONFIRMATION_COUNTDOWN
		int frequency = 800 + (800.0f * frames_since_fall) / FALL_CONFIRMATION_COUNTDOWN;
		if (frames_since_fall % 4 == 0) {
			tone(GPIO_BUZZER, frequency, 90);
		}
	} else if (frames_since_fall == FALL_CONFIRMATION_COUNTDOWN) {
		tone(GPIO_BUZZER, 3000, 500);
		// Fallen without cancelling
		LOG("Fall confirmed");

		NimBLEAdvertisementData advertisementData;
		pAdvertising->start();
		LOG("Started advertising fall status");
	} else if (frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// Keep advertising fall status
		if (frames_since_fall % 12 == 0) {
			tone(GPIO_BUZZER, 700, 500);
		}
	} else if (frames_since_fall > FALL_REPORTING_COUNTDOWN && pAdvertising->isAdvertising()) {
		pAdvertising->stop();
		LOG("Stopped advertising fall status");
	}

	unsigned long currentMillis = millis();
	unsigned long deltaTime = currentMillis - previousMillis;
	previousMillis = currentMillis;
	AccelData accelData;
	DeviceIMU.device.update();
	DeviceIMU.device.getAccel(&accelData);

	float deltaTimeF = (float)deltaTime;
	float acc_x = accelData.accelX;
	float acc_y = accelData.accelY;
	float acc_z = accelData.accelZ;

	if (++windex >= CIRCULARBUFFER_SIZE) windex = 0;

	circular_buffer[windex * CBC + 0] = acc_x;
	circular_buffer[windex * CBC + 1] = acc_y;
	circular_buffer[windex * CBC + 2] = acc_z;
	circular_buffer[windex * CBC + 3] = 0;

	float inference_result = 0.0f;
	static std::vector<long long> fall_indices;

	int fi = (windex - DOG_RADIUS + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
	{
		// We are interested in calculating the metrics in the middle of the window, at windex fi
		float mag_acc = sqrtf(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
		supporting_circular_buffer[windex * SBC + 0] = mag_acc;

		static float acc_x_sum = 0.0f;
		static float acc_y_sum = 0.0f;
		static float acc_z_sum = 0.0f;
		static float angle_sum = 0.0f;

		int fiP8 = (fi + 8) % CIRCULARBUFFER_SIZE;
		int fiM12 = (fi - 12 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
		int fiP28 = (fi + 28) % CIRCULARBUFFER_SIZE;
		int fiM8 = (fi - 8 + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;

		acc_x_sum += circular_buffer[fiP28 * CBC + 0];
		acc_x_sum -= circular_buffer[fiM12 * CBC + 0];
		acc_y_sum += circular_buffer[fiP28 * CBC + 1];
		acc_y_sum -= circular_buffer[fiM12 * CBC + 1];
		acc_z_sum += circular_buffer[fiP28 * CBC + 2];
		acc_z_sum -= circular_buffer[fiM12 * CBC + 2];

		float acc_x_lp = acc_x_sum / 40.0f;
		float acc_y_lp = acc_y_sum / 40.0f;
		float acc_z_lp = acc_z_sum / 40.0f;

		float acc_xP8 = circular_buffer[fiP8 * CBC + 0];
		float acc_yP8 = circular_buffer[fiP8 * CBC + 1];
		float acc_zP8 = circular_buffer[fiP8 * CBC + 2];

		float dot_prod = acc_xP8 * acc_x_lp + acc_yP8 * acc_y_lp + acc_zP8 * acc_z_lp;

		float mag_acc_P8 = supporting_circular_buffer[fiP8 * SBC + 0];
		float mag_acc_lp = sqrtf(acc_x_lp * acc_x_lp + acc_y_lp * acc_y_lp + acc_z_lp * acc_z_lp);
		float cos_angle = dot_prod / (mag_acc_P8 * mag_acc_lp + 1e-6f);
		float angle = acosf(min(max(cos_angle, -1.0f), 1.0f));
		supporting_circular_buffer[fiP8 * SBC + 1] = angle;
		angle_sum += angle;
		angle_sum -= supporting_circular_buffer[fiM8 * SBC + 1];
		float angle_lp = angle_sum / 16.0f;

		float dog_value = 0.0f;
		for (int i = 0; i < DOG_SIZE; i++) {
			int index = (fi - DOG_RADIUS + i + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
			float mag_acc_i = supporting_circular_buffer[index * SBC + 0];
			dog_value += mag_acc_i * dog_kernel[i];
		}
		float final_feature = dog_value * angle_lp;
		circular_buffer[fi * CBC + 3] = final_feature;

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
			// tone(25, 2800, 200);
			// tone(25, 1800, 200);
			max_feature_current_fall = -1.0f;
			max_feature_index = 0;
		}
	}

	// The center index of the inference window, where the custom feature is available for the entire window
	long long inference_center_index = loop_count - (INFERENCE_WINDOW_SIZE / 2 + DOG_RADIUS);
	while (!fall_indices.empty() && fall_indices.front() < inference_center_index) {
		fall_indices.erase(fall_indices.begin());	 // Remove old fall indices (too old to be relevant)
	}

	if (!fall_indices.empty() && fall_indices.front() == inference_center_index && frames_since_fall > FALL_REPORTING_COUNTDOWN) {
		fall_indices.erase(fall_indices.begin());
		LOG("Triggering inference...");
		unsigned long t_start = millis();
		// Fall detected used naive method at this index, trigger inference
		int fall_start_windex = (windex - (DOG_RADIUS + INFERENCE_WINDOW_SIZE) + CIRCULARBUFFER_SIZE) % CIRCULARBUFFER_SIZE;
		LOG("Preparing input data from windex: " + String(fall_start_windex));
		int first_copy_size = min(CIRCULARBUFFER_SIZE - fall_start_windex, INFERENCE_WINDOW_SIZE);
		int second_copy_size = INFERENCE_WINDOW_SIZE - first_copy_size;
		unsigned int first_copy_size_bytes = first_copy_size * CBC * sizeof(float);
		unsigned int second_copy_size_bytes = second_copy_size * CBC * sizeof(float);

		memcpy(input->data.f, &circular_buffer[fall_start_windex * CBC], first_copy_size_bytes);
		LOG("Copied first part of size " + String(first_copy_size_bytes) + " bytes to address: " + String((unsigned long)input->data.f, HEX));
		if (second_copy_size > 0) memcpy(input->data.f + first_copy_size * CBC, &circular_buffer[0], second_copy_size_bytes);
		LOG("Copied second part of size " + String(second_copy_size_bytes) + " bytes to address: " + String((unsigned long)(input->data.f + first_copy_size * CBC), HEX));

		TfLiteStatus invoke_status = interpreter->Invoke();
		unsigned long t_end = millis();
		if (invoke_status != kTfLiteOk) {
			LOG("Invoke failed! oh no :(");
		} else {
			float inference_result = output->data.f[0];
			LOG("Inference result: " + String(inference_result, 6) + " (took " + String(t_end - t_start) + " ms)");
			if (inference_result > 0.5) {
				LOG("Fall detected!");
				frames_since_fall = 0;
				// tone(GPIO_BUZZER, 3000, 1000);
			} else {
				LOG("False alarm.");
				// tone(GPIO_BUZZER, 1000, 1000);
			}
		}
	}

	int button_value = digitalRead(GPIO_BUTTON) == HIGH ? 0 : 1;
	if (button_value == 1 && frames_since_fall < FALL_REPORTING_COUNTDOWN) {
		// If button is pressed, and we are in fall countdown, cancel the fall (though it may be too late if already reported)
		frames_since_fall = FALL_REPORTING_COUNTDOWN + 1;
		LOG("Fall cancelled by button press");
		tone(GPIO_BUZZER, 500, 800);
	}

	static unsigned long time_spent = 0;

	LOG(time_spent);
	time_spent = millis() - currentMillis;
	// LOG("time_spent:" + String(time_spent) + " ms");
	unsigned long time_to_next = LOOP_DELAY > time_spent ? LOOP_DELAY - time_spent : 0;
	loop_count++;

	delay(time_to_next);
}
#endif