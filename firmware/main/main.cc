// Nœud Edge TinyML (ESP32-S3, exécuté sous QEMU) — diagnostic d'un moteur à partir de fenêtres de 2 s.
//
// Au démarrage :
//   1. vérifie l'empreinte SHA-256 du modèle (un modèle altéré n'est jamais exécuté) ;
//   2. charge le modèle INT8 dans TFLite Micro et vérifie ses paramètres de quantification ;
//   3. rejoue les fenêtres de référence exportées par Python et compare les sorties INT8 (parité C/Python) ;
//   4. mesure le nombre de cycles par inférence (indicatif sous QEMU, réel sur la carte).
// Ensuite, mode « logiciel dans la boucle » : chaque ligne « W,v1,…,v100 » reçue sur l'UART est classée et
// la réponse « R,<classe>,<5 logits INT8>,<cycles> » est renvoyée (voir edge/qemu_bridge.py).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "mbedtls/sha256.h"
#include "model_data.h"
#include "model_params.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char *TAG = "edge";

namespace {
constexpr int kArenaSize = 8 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_in = nullptr, *g_out = nullptr;
}  // namespace

static bool model_integrity_ok(void)
{
    uint8_t digest[32];
    if (mbedtls_sha256(g_model_data, g_model_data_len, digest, 0) != 0) return false;
    uint8_t diff = 0;
    for (int i = 0; i < 32; i++) diff |= digest[i] ^ EDGE_MODEL_SHA256[i];  // temps constant
    return diff == 0;
}

static bool model_init(void)
{
    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return false;
    static tflite::MicroMutableOpResolver<1> resolver;   // MLP : FULLY_CONNECTED uniquement (ReLU fusionnée)
    resolver.AddFullyConnected();
    static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) return false;
    g_interp = &interp;
    g_in = interp.input(0);
    g_out = interp.output(0);
    if (g_in->type != kTfLiteInt8 || g_out->type != kTfLiteInt8) return false;
    if (fabsf(g_in->params.scale - EDGE_IN_SCALE) > 1e-6f * EDGE_IN_SCALE || g_in->params.zero_point != EDGE_IN_ZP)
        return false;
    ESP_LOGI(TAG, "Modèle : %u octets | arène TFLM utilisée : %u / %d octets", g_model_data_len,
             (unsigned)interp.arena_used_bytes(), kArenaSize);
    return true;
}

// raw : fenêtre brute W x CH (ordre : échantillon puis capteur), identique au Normalizer Python
static int classify(const float *raw, int8_t out[EDGE_N_CLASSES], uint32_t *cycles)
{
    const float s = g_in->params.scale;
    const int zp = g_in->params.zero_point;
    for (int k = 0; k < EDGE_N_IN; k++) {
        const int c = k % EDGE_CH;
        const float xn = (raw[k] - EDGE_MEAN[c]) / EDGE_STD[c];
        long q = lroundf(xn / s) + zp;
        g_in->data.int8[k] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    const uint32_t c0 = esp_cpu_get_cycle_count();
    g_interp->Invoke();
    *cycles = esp_cpu_get_cycle_count() - c0;
    int best = 0;
    for (int k = 0; k < EDGE_N_CLASSES; k++) {
        out[k] = g_out->data.int8[k];
        if (out[k] > out[best]) best = k;
    }
    return best;
}

static bool parity_selftest(void)
{
    bool ok = true;
    int8_t out[EDGE_N_CLASSES];
    uint32_t cyc = 0;
    for (int g = 0; g < EDGE_N_GOLDEN; g++) {
        const int cls = classify(EDGE_GOLDEN_RAW[g], out, &cyc);
        int maxdiff = 0;
        for (int k = 0; k < EDGE_N_CLASSES; k++) maxdiff = fmax(maxdiff, abs(out[k] - EDGE_GOLDEN_OUT[g][k]));
        const bool pass = maxdiff <= 1;  // tolérance d'un pas de quantification (arrondis float32)
        ok &= pass;
        ESP_LOGI(TAG, "Parité %d : classe=%d  écart max=%d pas INT8  %lu cycles  %s", g, cls, maxdiff,
                 (unsigned long)cyc, pass ? "OK" : "ÉCHEC");
    }
    return ok;
}

static void serial_loop(void)
{
    uart_driver_install(UART_NUM_0, 4096, 0, 0, nullptr, 0);
    static char line[2048];
    static float raw[EDGE_N_IN];
    int len = 0;
    printf("PRET\n");
    for (;;) {
        uint8_t ch;
        if (uart_read_bytes(UART_NUM_0, &ch, 1, portMAX_DELAY) != 1) continue;
        if (ch != '\n') {
            if (len < (int)sizeof(line) - 1) line[len++] = (char)ch;
            continue;
        }
        line[len] = 0;
        len = 0;
        if (line[0] != 'W' || line[1] != ',') continue;
        char *p = line + 2;
        int n = 0;
        while (n < EDGE_N_IN && *p) {
            raw[n++] = strtof(p, &p);
            if (*p == ',') p++;
        }
        if (n != EDGE_N_IN) { printf("E,%d\n", n); continue; }
        int8_t out[EDGE_N_CLASSES];
        uint32_t cyc;
        const int cls = classify(raw, out, &cyc);
        printf("R,%d,%d,%d,%d,%d,%d,%lu\n", cls, out[0], out[1], out[2], out[3], out[4], (unsigned long)cyc);
    }
}

extern "C" void app_main(void)
{
    if (!model_integrity_ok()) {
        printf("TAMPER,model_sha256_mismatch\n");
        ESP_LOGE(TAG, "Empreinte du modèle non conforme : détecteur désactivé");
        return;
    }
    ESP_LOGI(TAG, "Empreinte SHA-256 du modèle : conforme");
    if (!model_init()) {
        ESP_LOGE(TAG, "Initialisation TFLM impossible");
        return;
    }
    if (!parity_selftest()) ESP_LOGE(TAG, "Auto-test de parité C/Python en échec");
    serial_loop();
}
