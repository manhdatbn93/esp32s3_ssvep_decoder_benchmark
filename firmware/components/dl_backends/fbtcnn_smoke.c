/*
 * FBtCNN smoke backend for TFLite Micro.
 *
 * Preprocessing: raw [512,8] → 4-band Butterworth filtfilt → [8,512,4]
 * fed to TFLite NHWC [1,8,512,4].  Both f32 and i8 TFLite models share
 * the same float32 preprocessing; the tflite_backend wrapper handles
 * quantization at the model I/O boundary for int8.
 */
#include "tflite_backend.h"
#include "fbtcnn_preprocess.h"
#include "smoke_vectors_fbtcnn.h"

#include <stdlib.h>

#define FBTCNN_N_CLASSES 12
#define FBTCNN_ARENA_SIZE (768 * 1024)

/*
 * Preprocessing output buffer (shared between f32/i8, never concurrent).
 * 16384 floats = 64 KB — allocated from PSRAM via heap_caps_malloc
 * at init time, freed at cleanup.
 */
static float *s_fbtcnn_features = NULL;

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_log.h"
static const char *TAG_FBTCNN = "fbtcnn";
#endif

static float *alloc_feature_buf(void)
{
    float *buf = NULL;
#ifdef ESP_PLATFORM
    buf = (float *)heap_caps_malloc(FBTCNN_FEATURE_LEN * sizeof(float),
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (buf) return buf;
#endif
    buf = (float *)malloc(FBTCNN_FEATURE_LEN * sizeof(float));
    return buf;
}

#ifdef SMOKE_ENABLE_FBTCNN_INT8
extern const uint8_t fbtcnn_int8_tflite_start[] asm("_binary_fbtcnn_int8_tflite_start");
extern const uint8_t fbtcnn_int8_tflite_end[]   asm("_binary_fbtcnn_int8_tflite_end");
#endif

#ifdef SMOKE_ENABLE_FBTCNN_F32
extern const uint8_t fbtcnn_float32_tflite_start[] asm("_binary_fbtcnn_float32_tflite_start");
extern const uint8_t fbtcnn_float32_tflite_end[]   asm("_binary_fbtcnn_float32_tflite_end");
#endif

#if defined(SMOKE_ENABLE_FBTCNN_INT8) || defined(SMOKE_ENABLE_FBTCNN_F32)
static tflite_ctx_t *s_ctx = NULL;

#ifdef ESP_PLATFORM
static void log_arena_profile(const char *label) {
    tflm_arena_profile_t p;
    if (tflite_ctx_get_arena_profile(s_ctx, &p) != 0) return;
    ESP_LOGI(TAG_FBTCNN,
             "TFLM_ARENA backend=%s cap=%u used=%u headroom=%u region=%s path=%s raw=0x%08x aligned=0x%08x "
             "int_before=%u int_after=%u spiram_before=%u spiram_after=%u used_avail=%d",
             label ? label : "FBtCNN",
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

/* ── FBtCNN int8 ────────────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_FBTCNN_INT8
int fbtcnn_i8_smoke_init(void) {
    s_fbtcnn_features = alloc_feature_buf();
    if (!s_fbtcnn_features) return -1;
    size_t len = fbtcnn_int8_tflite_end - fbtcnn_int8_tflite_start;
    s_ctx = tflite_ctx_create(fbtcnn_int8_tflite_start, len, FBTCNN_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("FBtCNN-int8");
#endif
    return s_ctx ? 0 : -1;
}

int fbtcnn_i8_smoke_run(const float *input_tc, float *scores_out) {
    fbtcnn_preprocess(input_tc, s_fbtcnn_features);
    return tflite_ctx_invoke(s_ctx, s_fbtcnn_features, FBTCNN_FEATURE_LEN,
                             scores_out, FBTCNN_N_CLASSES);
}

void fbtcnn_i8_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
    free(s_fbtcnn_features); s_fbtcnn_features = NULL;
}

const int          fbtcnn_i8_smoke_n_vectors = SMOKE_FBTCNN_N_VECTORS;
const float *const fbtcnn_i8_smoke_vectors[] = {
    SMOKE_FBTCNN_VEC0, SMOKE_FBTCNN_VEC1, SMOKE_FBTCNN_VEC2,
    SMOKE_FBTCNN_VEC3, SMOKE_FBTCNN_VEC4
};
const int fbtcnn_i8_smoke_expected_argmax[5] = { 6, 10, 0, 11, 9 };
const int   fbtcnn_i8_smoke_has_expected_scores = SMOKE_FBTCNN_HAS_EXPECTED_SCORES;
const float *fbtcnn_i8_smoke_expected_scores_flat = &SMOKE_FBTCNN_EXPECTED_SCORES[0][0];
#endif

/* ── FBtCNN float32 ─────────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_FBTCNN_F32
int fbtcnn_f32_smoke_init(void) {
    s_fbtcnn_features = alloc_feature_buf();
    if (!s_fbtcnn_features) return -1;
    size_t len = fbtcnn_float32_tflite_end - fbtcnn_float32_tflite_start;
    s_ctx = tflite_ctx_create(fbtcnn_float32_tflite_start, len, FBTCNN_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("FBtCNN-f32");
#endif
    return s_ctx ? 0 : -1;
}

int fbtcnn_f32_smoke_run(const float *input_tc, float *scores_out) {
    fbtcnn_preprocess(input_tc, s_fbtcnn_features);
    return tflite_ctx_invoke(s_ctx, s_fbtcnn_features, FBTCNN_FEATURE_LEN,
                             scores_out, FBTCNN_N_CLASSES);
}

void fbtcnn_f32_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
    free(s_fbtcnn_features); s_fbtcnn_features = NULL;
}

const int          fbtcnn_f32_smoke_n_vectors = SMOKE_FBTCNN_N_VECTORS;
const float *const fbtcnn_f32_smoke_vectors[] = {
    SMOKE_FBTCNN_VEC0, SMOKE_FBTCNN_VEC1, SMOKE_FBTCNN_VEC2,
    SMOKE_FBTCNN_VEC3, SMOKE_FBTCNN_VEC4
};
const int fbtcnn_f32_smoke_expected_argmax[5] = { 6, 10, 0, 11, 9 };
const int   fbtcnn_f32_smoke_has_expected_scores = SMOKE_FBTCNN_HAS_EXPECTED_SCORES;
const float *fbtcnn_f32_smoke_expected_scores_flat = &SMOKE_FBTCNN_EXPECTED_SCORES[0][0];
#endif
