/*
 * Shared TFLite Micro backend utilities.
 *
 * Provides a lightweight wrapper around TFLite Micro interpreter setup
 * so individual model backends only need to specify model data, arena
 * size, and preprocessing.
 */
#ifndef TFLITE_BACKEND_H
#define TFLITE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tflite_ctx tflite_ctx_t;

typedef struct {
    size_t arena_capacity_bytes;
    size_t arena_used_bytes;
    size_t arena_headroom_bytes;
    uintptr_t arena_raw_addr;
    uintptr_t arena_aligned_addr;
    const char *arena_region;      /* "PSRAM", "INTERNAL", "UNKNOWN" */
    const char *alloc_path;        /* "SPIRAM", "GENERIC_8BIT", "MALLOC", "UNKNOWN" */
    size_t heap_int_before_alloc;
    size_t heap_int_after_alloc;
    size_t heap_spiram_before_alloc;
    size_t heap_spiram_after_alloc;
    int arena_used_bytes_available;
} tflm_arena_profile_t;

/*
 * Create a TFLite Micro context from an embedded model flatbuffer.
 * arena_size: tensor arena in bytes (caller chooses based on model).
 * Returns NULL on failure.
 */
tflite_ctx_t *tflite_ctx_create(const uint8_t *model_data, size_t model_len,
                                size_t arena_size);

/*
 * Run inference. input/output are float32 regardless of internal quantization
 * (the interpreter handles dequantize/quantize at IO boundaries).
 */
int tflite_ctx_invoke(tflite_ctx_t *ctx,
                      const float *input, size_t input_len,
                      float *output, size_t output_len);

/* Print diagnostic info: arena alignment, tensor types, scales, zero-points. */
void tflite_ctx_log_info(tflite_ctx_t *ctx, const char *label);

/*
 * Query arena profiling info captured during context creation.
 * Returns 0 on success, non-zero on invalid args.
 */
int tflite_ctx_get_arena_profile(tflite_ctx_t *ctx, tflm_arena_profile_t *out);

/* Returns TfLiteType enum values for input/output tensor 0; -1 if unavailable. */
int tflite_ctx_get_io_dtypes(tflite_ctx_t *ctx, int *in_dtype, int *out_dtype);

void tflite_ctx_destroy(tflite_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_BACKEND_H */
