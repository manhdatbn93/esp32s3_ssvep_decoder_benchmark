/*
 * EEGNet smoke backend for TFLite Micro.
 *
 * Preprocessing: raw [512,8] → transpose to [8,512] for onnx2tf NHWC model.
 * No FFT or filter bank needed.
 *
 * Two variants: float32 and int8 (selected at compile time).
 */
#include "tflite_backend.h"
#include "model_identity.h"
#include "eegnet_model_identity.h"
#include "smoke_vectors_eegnet.h"
#include <string.h>

#include <inttypes.h>

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char *TAG_EN = "eegnet";
#endif

#define EEGNET_T 512
#define EEGNET_C 8
#define EEGNET_N_CLASSES 12
/* Int8 graphs use more scratch (quantized intermediates); 256 KiB was marginal. */
#if defined(SMOKE_ENABLE_EEGNET_INT8)
#define EEGNET_ARENA_SIZE (384 * 1024)
#else
#define EEGNET_ARENA_SIZE (256 * 1024)
#endif

#ifdef SMOKE_ENABLE_EEGNET_F32
extern const uint8_t eegnet_float32_tflite_start[] asm("_binary_eegnet_float32_tflite_start");
extern const uint8_t eegnet_float32_tflite_end[]   asm("_binary_eegnet_float32_tflite_end");
#endif

#ifdef SMOKE_ENABLE_EEGNET_INT8
extern const uint8_t eegnet_int8_tflite_start[] asm("_binary_eegnet_int8_tflite_start");
extern const uint8_t eegnet_int8_tflite_end[]   asm("_binary_eegnet_int8_tflite_end");
#endif

#if defined(SMOKE_ENABLE_EEGNET_F32) || defined(SMOKE_ENABLE_EEGNET_INT8)
static tflite_ctx_t *s_ctx = NULL;
static const uint8_t *s_model_start = NULL;
static size_t s_model_len = 0;
static int s_in_dtype = -1;
static int s_out_dtype = -1;

static void fill_blob_identity(eegnet_model_identity_t *out,
                               const uint8_t *start, size_t len,
                               const char *symbol, const char *backend) {
    if (!out || !start || len == 0) return;
    memset(out, 0, sizeof(*out));
    out->data = start;
    out->len = len;
    out->crc32 = model_identity_crc32(start, len);
    size_t n16 = len >= 16 ? 16 : len;
    memcpy(out->first16, start, n16);
    if (len >= 16)
        memcpy(out->last16, start + len - 16, 16);
    else
        memcpy(out->last16, start, n16);
    out->symbol = symbol;
    out->actual_backend = backend;
    if (s_ctx) {
        tflite_ctx_get_io_dtypes(s_ctx, &s_in_dtype, &s_out_dtype);
        out->input_dtype = s_in_dtype;
        out->output_dtype = s_out_dtype;
    }
}

static const char *tflite_dtype_name(int t) {
    switch (t) {
    case 1: return "float32";
    case 9: return "int8";
    default: return "?";
    }
}

static void log_fw_model_identity(const char *requested_backend,
                                  const eegnet_model_identity_t *id) {
#ifdef ESP_PLATFORM
    char fhex[33], lhex[33];
    model_identity_hex16(id->data, id->len, 0, fhex, sizeof(fhex));
    model_identity_hex16(id->data, id->len, 1, lhex, sizeof(lhex));
    ESP_LOGI(TAG_EN, "[fw:model] requested_backend=%s", requested_backend ? requested_backend : "?");
    ESP_LOGI(TAG_EN, "[fw:model] actual_backend=%s", id->actual_backend ? id->actual_backend : "?");
    ESP_LOGI(TAG_EN, "[fw:model] symbol=%s", id->symbol ? id->symbol : "?");
    ESP_LOGI(TAG_EN, "[fw:model] size=%u", (unsigned)id->len);
    ESP_LOGI(TAG_EN, "[fw:model] crc32=0x%08" PRIx32, (uint32_t)id->crc32);
    ESP_LOGI(TAG_EN, "[fw:model] first16=%s", fhex);
    ESP_LOGI(TAG_EN, "[fw:model] last16=%s", lhex);
    ESP_LOGI(TAG_EN, "[fw:model] input_dtype=%s", tflite_dtype_name(id->input_dtype));
    ESP_LOGI(TAG_EN, "[fw:model] output_dtype=%s", tflite_dtype_name(id->output_dtype));
#else
    (void)requested_backend;
    (void)id;
#endif
}

static float s_transposed[EEGNET_C * EEGNET_T];

static void transpose_tc_to_ct(const float *tc, float *ct, int T, int C) {
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            ct[c * T + t] = tc[t * C + c];
}

#ifdef ESP_PLATFORM
static void log_arena_profile(const char *label) {
    tflm_arena_profile_t p;
    if (tflite_ctx_get_arena_profile(s_ctx, &p) != 0) return;
    ESP_LOGI(TAG_EN,
             "TFLM_ARENA backend=%s cap=%u used=%u headroom=%u region=%s path=%s raw=0x%08x aligned=0x%08x "
             "int_before=%u int_after=%u spiram_before=%u spiram_after=%u used_avail=%d",
             label ? label : "EEGNet",
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

#ifdef SMOKE_ENABLE_EEGNET_F32
int eegnet_f32_smoke_init(void) {
    s_model_start = eegnet_float32_tflite_start;
    s_model_len = (size_t)(eegnet_float32_tflite_end - eegnet_float32_tflite_start);
    s_ctx = tflite_ctx_create(s_model_start, s_model_len, EEGNET_ARENA_SIZE);
#ifdef ESP_PLATFORM
    if (s_ctx) {
        log_arena_profile("EEGNet-f32");
        eegnet_model_identity_t id;
        fill_blob_identity(&id, s_model_start, s_model_len,
                           "_binary_eegnet_float32_tflite_start", "eegnet_f32");
        log_fw_model_identity("eegnet_f32", &id);
    }
#endif
    return s_ctx ? 0 : -1;
}

int eegnet_f32_model_identity(eegnet_model_identity_t *out) {
    if (!out || !s_model_start || s_model_len == 0) return -1;
    fill_blob_identity(out, s_model_start, s_model_len,
                       "_binary_eegnet_float32_tflite_start", "eegnet_f32");
    return 0;
}

void eegnet_f32_log_fw_model_identity(const char *requested_backend) {
    eegnet_model_identity_t id;
    if (eegnet_f32_model_identity(&id) != 0) return;
    log_fw_model_identity(requested_backend, &id);
}
int eegnet_f32_smoke_run(const float *input_tc, float *scores_out) {
    transpose_tc_to_ct(input_tc, s_transposed, EEGNET_T, EEGNET_C);
    return tflite_ctx_invoke(s_ctx, s_transposed, EEGNET_C * EEGNET_T,
                             scores_out, EEGNET_N_CLASSES);
}

void eegnet_f32_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx);
    s_ctx = NULL;
}

const int          eegnet_f32_smoke_n_vectors = SMOKE_EEGNET_N_VECTORS;
const float *const eegnet_f32_smoke_vectors[] = {
    SMOKE_EEGNET_VEC0, SMOKE_EEGNET_VEC1, SMOKE_EEGNET_VEC2,
    SMOKE_EEGNET_VEC3, SMOKE_EEGNET_VEC4
};
const int eegnet_f32_smoke_expected_argmax[5] = { 11, 0, 0, 10, 1 };
const int   eegnet_f32_smoke_has_expected_scores = SMOKE_EEGNET_HAS_EXPECTED_SCORES;
const float *eegnet_f32_smoke_expected_scores_flat = &SMOKE_EEGNET_EXPECTED_SCORES[0][0];
#endif

#ifdef SMOKE_ENABLE_EEGNET_INT8
int eegnet_i8_smoke_init(void) {
    s_model_start = eegnet_int8_tflite_start;
    s_model_len = (size_t)(eegnet_int8_tflite_end - eegnet_int8_tflite_start);
#ifdef ESP_PLATFORM
    ESP_LOGI(TAG_EN, "int8 init: model=%u bytes, arena=%u bytes",
             (unsigned)s_model_len, (unsigned)EEGNET_ARENA_SIZE);
#endif
    s_ctx = tflite_ctx_create(s_model_start, s_model_len, EEGNET_ARENA_SIZE);
    if (s_ctx) {
        tflite_ctx_log_info(s_ctx, "EEGNet-i8");
#ifdef ESP_PLATFORM
        log_arena_profile("EEGNet-int8");
        eegnet_model_identity_t id;
        fill_blob_identity(&id, s_model_start, s_model_len,
                           "_binary_eegnet_int8_tflite_start", "eegnet_int8");
        log_fw_model_identity("eegnet_int8", &id);
#endif
    }
    return s_ctx ? 0 : -1;
}

int eegnet_i8_model_identity(eegnet_model_identity_t *out) {
    if (!out || !s_model_start || s_model_len == 0) return -1;
    fill_blob_identity(out, s_model_start, s_model_len,
                       "_binary_eegnet_int8_tflite_start", "eegnet_int8");
    return 0;
}

void eegnet_i8_log_fw_model_identity(const char *requested_backend) {
    eegnet_model_identity_t id;
    if (eegnet_i8_model_identity(&id) != 0) return;
    log_fw_model_identity(requested_backend, &id);
}

int eegnet_i8_smoke_run(const float *input_tc, float *scores_out) {
    transpose_tc_to_ct(input_tc, s_transposed, EEGNET_T, EEGNET_C);
    return tflite_ctx_invoke(s_ctx, s_transposed, EEGNET_C * EEGNET_T,
                             scores_out, EEGNET_N_CLASSES);
}

void eegnet_i8_smoke_cleanup(void) {
    tflite_ctx_destroy(s_ctx);
    s_ctx = NULL;
}

const int          eegnet_i8_smoke_n_vectors = SMOKE_EEGNET_N_VECTORS;
const float *const eegnet_i8_smoke_vectors[] = {
    SMOKE_EEGNET_VEC0, SMOKE_EEGNET_VEC1, SMOKE_EEGNET_VEC2,
    SMOKE_EEGNET_VEC3, SMOKE_EEGNET_VEC4
};
const int eegnet_i8_smoke_expected_argmax[5] = { 11, 0, 0, 10, 1 };
const int   eegnet_i8_smoke_has_expected_scores = SMOKE_EEGNET_HAS_EXPECTED_SCORES;
const float *eegnet_i8_smoke_expected_scores_flat = &SMOKE_EEGNET_EXPECTED_SCORES[0][0];
#endif

#endif /* SMOKE_ENABLE_EEGNET_F32 || SMOKE_ENABLE_EEGNET_INT8 */
