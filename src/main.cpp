#include <Arduino.h>

// #include <AsyncHTTPRequest_Generic.h>

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

int log_metrics = 0;
char server_host[] = "10.112.150.112";
int server_port = 8000;
char server_message[] = "POST /i_have_fallen HTTP/1.0";

int state_window[64] = {};
float data_window[64 * 8] = {};
int windex = 0;
int frames_since_fall = 100;
unsigned long previousMillis = 0;
WiFiClient client;
// AsyncHTTPRequest request;

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000 * 10;
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

	// Get input details
	Serial.println("Input details");
	Serial.println("Input type: " + String(input->type));
	Serial.println("Input dims: " + String(input->dims->size));
	for (int i = 0; i < input->dims->size; i++) {
		Serial.println(" Input dim " + String(i) + ": " + String(input->dims->data[i]));
	}
	Serial.println("Input scale: " + String(input->params.scale));
	Serial.println("Input zero_point: " + String(input->params.zero_point));

	// PeripheralIMU.setup(4, 5);	// SDA, SCL
	PeripheralIMU.setup(27, 26);	// SDA, SCL

	PeripheralWifi.connect(1);
	PeripheralWifi.waitForConnect(1);

	pinMode(GPIO_BUTTON, INPUT_PULLUP);	 // Button

	// pinMode(LED_BUILTIN, OUTPUT);
	// digitalWrite(LED_BUILTIN, HIGH);
	// pinMode(0, OUTPUT);
	// digitalWrite(0, HIGH);	// Wifi is on
	previousMillis = millis();
}

void loop() {
	unsigned long currentMillis = millis();
	unsigned long deltaTime = currentMillis - previousMillis;
	previousMillis = currentMillis;
	AccelData accelData;
	GyroData gyroData;
	PeripheralIMU.device.update();
	PeripheralIMU.device.getAccel(&accelData);
	PeripheralIMU.device.getGyro(&gyroData);

	float acc_x = accelData.accelX;
	float acc_y = accelData.accelY;
	float acc_z = accelData.accelZ;
	float gyro_x = gyroData.gyroX;
	float gyro_y = gyroData.gyroY;
	float gyro_z = gyroData.gyroZ;
	float acc_mag = sqrtf(pow(acc_x, 2) + pow(acc_y, 2) + pow(acc_z, 2));

	int weightless = acc_mag < 0.6;
	int peak = acc_mag > 2.0;
	int stable = acc_mag > 0.7 && acc_mag < 1.3;

	if (weightless) {
		state_window[windex] = 1;
	} else if (peak) {
		state_window[windex] = 2;
	} else if (stable) {
		state_window[windex] = 3;
	} else {
		state_window[windex] = 0;
	}

	int fallen = 0;
	if (frames_since_fall < 50) {
		frames_since_fall++;
		fallen = 1;
	} else {
		for (int j = 0; j < 1; j++) {
			int windex_t = windex;
			int stables_remaining = 4;
			for (int i = 0; i < 4; i++) {
				if (state_window[windex_t] == 3)
					stables_remaining--;
				else
					stables_remaining = 4;
				windex_t = (windex_t - 1 + 64) % 64;
				if (stables_remaining <= 0) break;
			}
			if (stables_remaining > 0) break;

			int peaks_remaining = 1;
			for (int i = 0; i < 9 + 1; i++) {
				if (state_window[windex_t] == 2)
					peaks_remaining--;
				else
					peaks_remaining = 1;
				windex_t = (windex_t - 1 + 64) % 64;
				if (peaks_remaining <= 0) break;
			}
			if (peaks_remaining > 0) break;

			int weightless_remaining = 3;
			for (int i = 0; i < 5 + 3; i++) {
				if (state_window[windex_t] == 1)
					weightless_remaining--;
				else
					weightless_remaining = 3;
				windex_t = (windex_t - 1 + 64) % 64;
				if (weightless_remaining <= 0) break;
			}
			if (weightless_remaining > 0) break;
			fallen = 1;
			frames_since_fall = 0;
		}
	}

	if (frames_since_fall == 0) {
		Serial.println("Fall detected!");
		// if (PeripheralWifi.isConnected() and client.connect(server_host, server_port)) {
		// 	Serial.println(server_message);
		// 	client.println(server_message);
		// 	client.println();
		// 	Serial.println("Sent fall message to server");
		// } else {
		// 	Serial.println("Connection to server failed");
		// }
	}
	if (fallen == 1) {
		int odd = frames_since_fall & 1;
		// digitalWrite(LED_BUILTIN, odd ? LOW : HIGH);
	}

	windex = (windex + 1) % 64;

	// digitalWrite(0, PeripheralWifi.isConnected() ? HIGH : LOW);

	int button_value = digitalRead(GPIO_BUTTON) == HIGH ? 0 : 1;

	if (log_metrics) {
		Serial.print(">time_ms:");
		Serial.println(currentMillis);
		Serial.print(">button:");
		Serial.println(button_value);

		Serial.print(">acc_mag:");
		Serial.println(acc_mag);

		Serial.print(">acc_x:");
		Serial.println(acc_x);
		Serial.print(">acc_y:");
		Serial.println(acc_y);
		Serial.print(">acc_z:");
		Serial.println(acc_z);

		Serial.println(">weightless:" + String(weightless));
		Serial.println(">peak:" + String(peak));
		Serial.println(">stable:" + String(stable));
		Serial.println(">fallen:" + String(fallen));
	}

	if (windex == 0) {
		Serial.println("-----");
		if (PeripheralWifi.isConnected() and client.connect(server_host, server_port)) {
			// Send entire contents of data_window in HTTP POST request
			client.println("POST /log_data HTTP/1.0");
			client.println("Content-Type: application/octet-stream");
			client.println("Content-Length: " + String(sizeof(data_window)));
			client.println();
			client.write((uint8_t*)data_window, sizeof(data_window));
			client.println();
			Serial.println("Sent data window to server");
			// get response from server
			// while (client.connected()) {
			// 	if (client.available()) {
			// 		String line = client.readStringUntil('\n');
			// 		Serial.println(line);
			// 	}
			// }
		} else {
			Serial.println("Connection to server failed");
		}
	}

	data_window[windex + (0 * 64)] = *((float*)(&currentMillis));
	data_window[windex + (1 * 64)] = *((float*)(&button_value));
	data_window[windex + (2 * 64)] = acc_x;
	data_window[windex + (3 * 64)] = acc_y;
	data_window[windex + (4 * 64)] = acc_z;
	data_window[windex + (5 * 64)] = gyro_x;
	data_window[windex + (6 * 64)] = gyro_y;
	data_window[windex + (7 * 64)] = gyro_z;

	for (int i = 0; i < 64 * 6; i++) {
		input->data.f[i] = data_window[i + (2 * 64)];
	}

	// // Run inference, and report any error
	// TfLiteStatus invoke_status = interpreter->Invoke();
	// if (invoke_status != kTfLiteOk) {
	// 	Serial.println("Invoke failed! oh no :(");
	// 	return;
	// }

	// float y = output->data.f[0];
	// Serial.println(y);
	unsigned long time_spent = millis() - currentMillis;
	unsigned long time_to_next = LOOP_DELAY > time_spent ? LOOP_DELAY - time_spent : 0;

	delay(time_to_next);
}