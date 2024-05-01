/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

#include <PDM.h>
#include <Cough_or_Not_detector_inferencing.h>

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

/** LED indication of classification */
static signed short redPin = 22;
static signed short greenPin = 23;
static signed short bluePin = 24;

/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    pinMode(LED_BUILTIN, OUTPUT); // Setup the built-in LED
    pinMode(redPin, OUTPUT); //set up built RGB
    pinMode(greenPin, OUTPUT);
    pinMode(bluePin, OUTPUT);

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
}

void setLed(signed short red, signed short green, signed short blue) {
    analogWrite(redPin, 255 - red);
    analogWrite(greenPin, 255- green);
    analogWrite(bluePin, 255 - blue);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    ei_printf("Starting inferencing in 2 seconds...\n");
    delay(2000);
    ei_printf("Recording...\n");

    if (!microphone_inference_record()) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    ei_printf("Recording done\n");
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;

    ei_impulse_result_t result;
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    // Control the built-in LED and RGB LEDs based on the highest probability
    float cough_prob = result.classification[0].value;
    float sneeze_prob = result.classification[1].value;
    float background_noise_prob = result.classification[2].value;

    if (cough_prob > sneeze_prob && cough_prob > background_noise_prob && cough_prob > 0.5) {
        digitalWrite(LED_BUILTIN, HIGH); // Turn on the LED for cough
        for (int i = 0; i < 3; i++) {
            setLed(255, 0, 0); // Blink red LED for cough
            delay(100);
            //setLed(255, 255, 255); // LEDs off
            //delay(100);
        }
    } else if (sneeze_prob > cough_prob && sneeze_prob > background_noise_prob && sneeze_prob > 0.5) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100); digitalWrite(LED_BUILTIN, LOW);
        delay(100); digitalWrite(LED_BUILTIN, HIGH);
        delay(100); digitalWrite(LED_BUILTIN, LOW);
        for (int i = 0; i < 3; i++) {
            setLed(0, 0, 255); // Blink blue LED for sneeze
            delay(100);
            //setLed(255, 255, 255); // LEDs off
            //delay(100);
        }
    } else if (background_noise_prob > cough_prob && background_noise_prob > sneeze_prob && background_noise_prob > 0.6) {
        digitalWrite(LED_BUILTIN, LOW); // Turn off the LED
        setLed(0, 255, 0); // Green LED on for background noise
        delay(100);
    } else {
        digitalWrite(LED_BUILTIN, LOW); // Turn off the LED
        setLed(0, 0, 0); // All LEDs off
    }
}

/**
 * @brief      PDM buffer full callback
 */
static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    if (bytesAvailable > 0) {
        int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);
        for(int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];
            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 */
static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) {
        return false;
    }
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;
    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }
    PDM.setGain(127);
    return true;
}

/**
 * @brief      Wait on new data
 */
static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while (inference.buf_ready == 0) {
        delay(10);
    }
    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    return numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
