/*
 * CCNN smoke backend for TFLite Micro.
 *
 * Preprocessing: raw [512,8] → Goertzel DFT → spectrum features [8,220]
 * fed to TFLite NHWC [1,8,220,1].  Both f32 and i8 TFLite models share
 * the same float32 preprocessing; the tflite_backend wrapper handles
 * quantization at the model I/O boundary for int8.
 */
#include "tflite_backend.h"
#include "ccnn_preprocess.h"
#include "smoke_vectors_ccnn.h"

#define CCNN_N_CLASSES 12
#define CCNN_ARENA_SIZE (512 * 1024)

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char *TAG_CCNN = "ccnn";
#endif

static float s_ccnn_features[CCNN_FEATURE_LEN];

#ifdef SMOKE_ENABLE_CCNN_F32
extern const uint8_t ccnn_float32_tflite_start[] asm("_binary_ccnn_float32_tflite_start");
extern const uint8_t ccnn_float32_tflite_end[]   asm("_binary_ccnn_float32_tflite_end");
#endif

#ifdef SMOKE_ENABLE_CCNN_INT8
extern const uint8_t ccnn_int8_tflite_start[] asm("_binary_ccnn_int8_tflite_start");
extern const uint8_t ccnn_int8_tflite_end[]   asm("_binary_ccnn_int8_tflite_end");
#endif

#if defined(SMOKE_ENABLE_CCNN_F32) || defined(SMOKE_ENABLE_CCNN_INT8)
static tflite_ctx_t *s_ctx = NULL;

#ifdef ESP_PLATFORM
static void log_arena_profile(const char *label) {
    tflm_arena_profile_t p;
    if (tflite_ctx_get_arena_profile(s_ctx, &p) != 0) return;
    ESP_LOGI(TAG_CCNN,
             "TFLM_ARENA backend=%s cap=%u used=%u headroom=%u region=%s path=%s raw=0x%08x aligned=0x%08x "
             "int_before=%u int_after=%u spiram_before=%u spiram_after=%u used_avail=%d",
             label ? label : "CCNN",
             (unsigned)p.arena_capacity_bytes,
             (unsigned)p.arena_used_bytes,
             (unsigned)p.arena_headroom_bytes,
             p.arena_region ? p.arena_region : "UNKNOWN",
             p.alloc_path ? p.alloc_path : "UNKNOWN",
             (unsigned)p.arena_raw_addr,
             (unsigned)p.arena_aligned_addr,
             (unsigned)p.heap_int_before_alloc,
             (unsigned)p.heap_int_after_alloc,
             (unsigned)p.heap_spiram_before_alloc,
             (unsigned)p.heap_spiram_after_alloc,
             p.arena_used_bytes_available);
}
#endif
#endif

/* ── CCNN float32 ───────────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_CCNN_F32
int ccnn_f32_smoke_init(void) {
    ccnn_preprocess_init();
    size_t len = ccnn_float32_tflite_end - ccnn_float32_tflite_start;
    s_ctx = tflite_ctx_create(ccnn_float32_tflite_start, len, CCNN_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("CCNN-f32");
#endif
    return s_ctx ? 0 : -1;
}

int ccnn_f32_smoke_run(const float *input_tc, float *scores_out) {
    ccnn_preprocess(input_tc, s_ccnn_features);
    return tflite_ctx_invoke(s_ctx, s_ccnn_features, CCNN_FEATURE_LEN,
                             scores_out, CCNN_N_CLASSES);
}

void ccnn_f32_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
}

const int          ccnn_f32_smoke_n_vectors = SMOKE_CCNN_N_VECTORS;
const float *const ccnn_f32_smoke_vectors[] = {
    SMOKE_CCNN_VEC0, SMOKE_CCNN_VEC1, SMOKE_CCNN_VEC2,
    SMOKE_CCNN_VEC3, SMOKE_CCNN_VEC4
};
const int ccnn_f32_smoke_expected_argmax[5] = { 10, 4, 0, 11, 9 };
const int   ccnn_f32_smoke_has_expected_scores = SMOKE_CCNN_HAS_EXPECTED_SCORES;
const float *ccnn_f32_smoke_expected_scores_flat = &SMOKE_CCNN_EXPECTED_SCORES[0][0];
#endif

/* ── CCNN int8 ──────────────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_CCNN_INT8
int ccnn_i8_smoke_init(void) {
    ccnn_preprocess_init();
    size_t len = ccnn_int8_tflite_end - ccnn_int8_tflite_start;
    s_ctx = tflite_ctx_create(ccnn_int8_tflite_start, len, CCNN_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("CCNN-int8");
#endif
    return s_ctx ? 0 : -1;
}

int ccnn_i8_smoke_run(const float *input_tc, float *scores_out) {
    ccnn_preprocess(input_tc, s_ccnn_features);
    return tflite_ctx_invoke(s_ctx, s_ccnn_features, CCNN_FEATURE_LEN,
                             scores_out, CCNN_N_CLASSES);
}

void ccnn_i8_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
}

const int          ccnn_i8_smoke_n_vectors = SMOKE_CCNN_N_VECTORS;
const float *const ccnn_i8_smoke_vectors[] = {
    SMOKE_CCNN_VEC0, SMOKE_CCNN_VEC1, SMOKE_CCNN_VEC2,
    SMOKE_CCNN_VEC3, SMOKE_CCNN_VEC4
};
const int ccnn_i8_smoke_expected_argmax[5] = { 10, 4, 0, 11, 9 };
const int   ccnn_i8_smoke_has_expected_scores = SMOKE_CCNN_HAS_EXPECTED_SCORES;
const float *ccnn_i8_smoke_expected_scores_flat = &SMOKE_CCNN_EXPECTED_SCORES[0][0];
#endif
