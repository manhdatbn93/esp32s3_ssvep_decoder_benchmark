/*
 * ConvCA smoke backend for TFLite Micro.
 *
 * Preprocessing: raw [512,8] → feed directly (model used -kat, keeps layout).
 * Templates are baked into the model as constant buffers.
 *
 * float32 is the recommended deploy path. int8 is diagnostic only.
 */
#include "tflite_backend.h"
#include "smoke_vectors_convca.h"
#include <string.h>

#define CONVCA_T 512
#define CONVCA_C 8
#define CONVCA_N_CLASSES 12
#define CONVCA_ARENA_SIZE (512 * 1024)

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char *TAG_CONVCA = "convca";
#endif

#ifdef SMOKE_ENABLE_CONVCA_F32
extern const uint8_t convca_float32_tflite_start[] asm("_binary_convca_float32_tflite_start");
extern const uint8_t convca_float32_tflite_end[]   asm("_binary_convca_float32_tflite_end");
#endif

#ifdef SMOKE_ENABLE_CONVCA_INT8
extern const uint8_t convca_int8_tflite_start[] asm("_binary_convca_int8_tflite_start");
extern const uint8_t convca_int8_tflite_end[]   asm("_binary_convca_int8_tflite_end");
#endif

#if defined(SMOKE_ENABLE_CONVCA_F32) || defined(SMOKE_ENABLE_CONVCA_INT8)
static tflite_ctx_t *s_ctx = NULL;

#ifdef ESP_PLATFORM
static void log_arena_profile(const char *label) {
    tflm_arena_profile_t p;
    if (tflite_ctx_get_arena_profile(s_ctx, &p) != 0) return;
    ESP_LOGI(TAG_CONVCA,
             "TFLM_ARENA backend=%s cap=%u used=%u headroom=%u region=%s path=%s raw=0x%08x aligned=0x%08x "
             "int_before=%u int_after=%u spiram_before=%u spiram_after=%u used_avail=%d",
             label ? label : "ConvCA",
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

#ifdef SMOKE_ENABLE_CONVCA_F32
int convca_f32_smoke_init(void) {
    size_t len = convca_float32_tflite_end - convca_float32_tflite_start;
    s_ctx = tflite_ctx_create(convca_float32_tflite_start, len, CONVCA_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("ConvCA-f32");
#endif
    return s_ctx ? 0 : -1;
}

int convca_f32_smoke_run(const float *input_tc, float *scores_out) {
    return tflite_ctx_invoke(s_ctx, input_tc, CONVCA_T * CONVCA_C,
                             scores_out, CONVCA_N_CLASSES);
}

void convca_f32_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
}

const int          convca_f32_smoke_n_vectors = SMOKE_CONVCA_N_VECTORS;
const float *const convca_f32_smoke_vectors[] = {
    SMOKE_CONVCA_VEC0, SMOKE_CONVCA_VEC1, SMOKE_CONVCA_VEC2,
    SMOKE_CONVCA_VEC3, SMOKE_CONVCA_VEC4
};
const int convca_f32_smoke_expected_argmax[5] = { 3, 7, 0, 11, 4 };
const int   convca_f32_smoke_has_expected_scores = SMOKE_CONVCA_HAS_EXPECTED_SCORES;
const float *convca_f32_smoke_expected_scores_flat = &SMOKE_CONVCA_EXPECTED_SCORES[0][0];
#endif

#ifdef SMOKE_ENABLE_CONVCA_INT8
int convca_i8_smoke_init(void) {
    size_t len = convca_int8_tflite_end - convca_int8_tflite_start;
    s_ctx = tflite_ctx_create(convca_int8_tflite_start, len, CONVCA_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) log_arena_profile("ConvCA-int8");
#endif
    return s_ctx ? 0 : -1;
}

int convca_i8_smoke_run(const float *input_tc, float *scores_out) {
    return tflite_ctx_invoke(s_ctx, input_tc, CONVCA_T * CONVCA_C,
                             scores_out, CONVCA_N_CLASSES);
}

void convca_i8_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx); s_ctx = NULL;
}

const int          convca_i8_smoke_n_vectors = SMOKE_CONVCA_N_VECTORS;
const float *const convca_i8_smoke_vectors[] = {
    SMOKE_CONVCA_VEC0, SMOKE_CONVCA_VEC1, SMOKE_CONVCA_VEC2,
    SMOKE_CONVCA_VEC3, SMOKE_CONVCA_VEC4
};
const int convca_i8_smoke_expected_argmax[5] = { 3, 7, 0, 11, 4 };
const int   convca_i8_smoke_has_expected_scores = SMOKE_CONVCA_HAS_EXPECTED_SCORES;
const float *convca_i8_smoke_expected_scores_flat = &SMOKE_CONVCA_EXPECTED_SCORES[0][0];
#endif
