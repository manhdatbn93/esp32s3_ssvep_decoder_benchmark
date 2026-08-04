/*
 * TFLite Micro backend wrapper for ESP32-S3.
 *
 * This file is compiled only when SMOKE_ENABLE_DL is set.
 * It wraps the esp-tflite-micro C++ API behind a plain-C interface
 * so model-specific backends (eegnet_smoke.c, etc.) stay in C.
 */

/* esp-tflite-micro provides a C-compatible API via these headers */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tflite_backend.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_log.h"
static const char *TAG = "tflm";
#define TFLM_LOGI(fmt, ...) ESP_LOGI(TAG, fmt, ##__VA_ARGS__)
#else
#define TFLM_LOGI(fmt, ...) printf("[tflm] " fmt "\n", ##__VA_ARGS__)
#endif

/*
 * Maximum number of TFLite ops needed by any of our models.
 * EEGNet: ~11, CCNN: ~6, ConvCA: ~10, FBtCNN: ~9.
 */
#define MAX_OPS 20

struct tflite_ctx {
    /* Tensor arena must be 16-byte aligned for TFLite Micro. arena_raw is the
     * pointer returned by malloc/heap_caps (may be unaligned); free that. */
    uint8_t *arena;
    void *arena_raw;
    tflite::MicroInterpreter *interpreter;
    const tflite::Model *model;
    tflm_arena_profile_t profile;
};

/* MicroMutableOpResolver must be populated exactly once: AddConv2D() etc. append
 * op slots; calling tflite_ctx_create() again without a fresh resolver would
 * overflow MAX_OPS or leave the resolver in an inconsistent state. */
static tflite::MicroMutableOpResolver<MAX_OPS> s_resolver;
static bool s_resolver_ready = false;

static void ensure_resolver(void) {
    if (s_resolver_ready) return;
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddFullyConnected();
    s_resolver.AddReshape();
    s_resolver.AddMean();
    s_resolver.AddPad();
    s_resolver.AddStridedSlice();
    s_resolver.AddAveragePool2D();
    s_resolver.AddTranspose();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
    s_resolver.AddMul();
    s_resolver.AddDiv();
    s_resolver.AddSqrt();
    s_resolver.AddSum();
    s_resolver.AddBatchMatMul();
    s_resolver.AddElu();
    s_resolver_ready = true;
}

static int s_invoke_count = 0;

/* Allocate (need + 15) bytes and return the first 16-byte-aligned address inside. */
static uint8_t *alloc_aligned_arena(size_t need, void **out_raw,
                                    const char **out_path,
                                    size_t *heap_int_before,
                                    size_t *heap_int_after,
                                    size_t *heap_spiram_before,
                                    size_t *heap_spiram_after) {
    const size_t pad = need + 16u;
    void *raw = nullptr;
    const char *path = "UNKNOWN";
#ifdef ESP_PLATFORM
    size_t int_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t spiram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#else
    size_t int_before = 0;
    size_t spiram_before = 0;
#endif
#ifdef ESP_PLATFORM
    raw = heap_caps_malloc(pad, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (raw) {
        path = "SPIRAM";
    } else {
        raw = heap_caps_malloc(pad, MALLOC_CAP_8BIT);
        if (raw) {
            path = "GENERIC_8BIT";
        }
    }
#endif
    if (!raw) {
        raw = malloc(pad);
        if (raw) path = "MALLOC";
    }
#ifdef ESP_PLATFORM
    size_t int_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t spiram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#else
    size_t int_after = 0;
    size_t spiram_after = 0;
#endif
    if (!raw) {
        *out_raw = nullptr;
        if (out_path) *out_path = path;
        if (heap_int_before) *heap_int_before = int_before;
        if (heap_int_after) *heap_int_after = int_after;
        if (heap_spiram_before) *heap_spiram_before = spiram_before;
        if (heap_spiram_after) *heap_spiram_after = spiram_after;
        return nullptr;
    }
    uintptr_t a = ((uintptr_t)raw + 15u) & ~(uintptr_t)15u;
    *out_raw = raw;
    if (out_path) *out_path = path;
    if (heap_int_before) *heap_int_before = int_before;
    if (heap_int_after) *heap_int_after = int_after;
    if (heap_spiram_before) *heap_spiram_before = spiram_before;
    if (heap_spiram_after) *heap_spiram_after = spiram_after;
    return (uint8_t *)a;
}

static const char *arena_region_from_ptr(const void *p) {
#ifdef ESP_PLATFORM
    if (!p) return "UNKNOWN";
    if (esp_ptr_external_ram(p)) return "PSRAM";
    if (esp_ptr_internal(p)) return "INTERNAL";
    return "UNKNOWN";
#else
    (void)p;
    return "UNKNOWN";
#endif
}

extern "C" tflite_ctx_t *tflite_ctx_create(const uint8_t *model_data,
                                            size_t model_len,
                                            size_t arena_size) {
    tflite_ctx *ctx = (tflite_ctx *)calloc(1, sizeof(tflite_ctx));
    if (!ctx) return NULL;

    memset(&ctx->profile, 0, sizeof(ctx->profile));
    const char *alloc_path = "UNKNOWN";
    size_t int_before = 0, int_after = 0, spiram_before = 0, spiram_after = 0;
    ctx->arena = alloc_aligned_arena(
        arena_size, &ctx->arena_raw, &alloc_path,
        &int_before, &int_after, &spiram_before, &spiram_after
    );
    if (!ctx->arena) {
        free(ctx);
        return NULL;
    }

    TFLM_LOGI("arena: raw=%p aligned=%p size=%u align_ok=%d resolver_was_ready=%d",
              ctx->arena_raw, (void *)ctx->arena, (unsigned)arena_size,
              (((uintptr_t)ctx->arena & 0xF) == 0), (int)s_resolver_ready);

    ctx->model = tflite::GetModel(model_data);
    if (!ctx->model) {
#ifdef ESP_PLATFORM
        heap_caps_free(ctx->arena_raw);
#else
        free(ctx->arena_raw);
#endif
        free(ctx);
        return NULL;
    }

    ensure_resolver();

    auto *interp = new tflite::MicroInterpreter(ctx->model, s_resolver,
                                                ctx->arena, arena_size);
    if (interp->AllocateTensors() != kTfLiteOk) {
        delete interp;
#ifdef ESP_PLATFORM
        heap_caps_free(ctx->arena_raw);
#else
        free(ctx->arena_raw);
#endif
        free(ctx);
        return NULL;
    }
    ctx->interpreter = interp;
    ctx->profile.arena_capacity_bytes = arena_size;
    ctx->profile.arena_raw_addr = (uintptr_t)ctx->arena_raw;
    ctx->profile.arena_aligned_addr = (uintptr_t)ctx->arena;
    ctx->profile.arena_region = arena_region_from_ptr(ctx->arena_raw);
    ctx->profile.alloc_path = alloc_path;
    ctx->profile.heap_int_before_alloc = int_before;
    ctx->profile.heap_int_after_alloc = int_after;
    ctx->profile.heap_spiram_before_alloc = spiram_before;
    ctx->profile.heap_spiram_after_alloc = spiram_after;
    ctx->profile.arena_used_bytes = interp->arena_used_bytes();
    ctx->profile.arena_used_bytes_available = 1;
    ctx->profile.arena_headroom_bytes =
        (ctx->profile.arena_used_bytes <= arena_size)
            ? (arena_size - ctx->profile.arena_used_bytes)
            : 0;
    s_invoke_count = 0;
    TFLM_LOGI(
        "TFLM_ARENA cap=%u used=%u headroom=%u raw=%p aligned=%p region=%s alloc_path=%s "
        "int_before=%u int_after=%u spiram_before=%u spiram_after=%u used_avail=%d",
        (unsigned)ctx->profile.arena_capacity_bytes,
        (unsigned)ctx->profile.arena_used_bytes,
        (unsigned)ctx->profile.arena_headroom_bytes,
        (void *)ctx->profile.arena_raw_addr,
        (void *)ctx->profile.arena_aligned_addr,
        ctx->profile.arena_region ? ctx->profile.arena_region : "UNKNOWN",
        ctx->profile.alloc_path ? ctx->profile.alloc_path : "UNKNOWN",
        (unsigned)ctx->profile.heap_int_before_alloc,
        (unsigned)ctx->profile.heap_int_after_alloc,
        (unsigned)ctx->profile.heap_spiram_before_alloc,
        (unsigned)ctx->profile.heap_spiram_after_alloc,
        ctx->profile.arena_used_bytes_available
    );
    TFLM_LOGI("ctx_create OK: interp=%p", (void *)interp);
    return (tflite_ctx_t *)ctx;
}

extern "C" void tflite_ctx_log_info(tflite_ctx_t *ctxp, const char *label) {
    tflite_ctx *ctx = (tflite_ctx *)ctxp;
    if (!ctx || !ctx->interpreter) {
        TFLM_LOGI("[%s] ctx NULL or no interpreter", label ? label : "?");
        return;
    }
    TfLiteTensor *in_t = ctx->interpreter->input(0);
    TfLiteTensor *out_t = ctx->interpreter->output(0);

    const char *type_name[] = {"none","f32","i32","u8","i64","str","bool","i16",
                               "complex64","i8","f16","f64","complex128","u64",
                               "resource","variant","u32","u16","i4","u4"};
    auto tname = [&](TfLiteType t) -> const char * {
        int ti = (int)t;
        if (ti >= 0 && ti < (int)(sizeof(type_name)/sizeof(type_name[0])))
            return type_name[ti];
        return "?";
    };

    TFLM_LOGI("[%s] input  type=%s(%d) bytes=%u scale=%.8f zp=%d",
              label, tname(in_t->type), (int)in_t->type,
              (unsigned)in_t->bytes,
              (double)in_t->params.scale, (int)in_t->params.zero_point);
    TFLM_LOGI("[%s] output type=%s(%d) bytes=%u scale=%.8f zp=%d",
              label, tname(out_t->type), (int)out_t->type,
              (unsigned)out_t->bytes,
              (double)out_t->params.scale, (int)out_t->params.zero_point);
    TFLM_LOGI("[%s] arena raw=%p aligned=%p align_ok=%d",
              label, ctx->arena_raw, (void *)ctx->arena,
              (((uintptr_t)ctx->arena & 0xF) == 0));
}

extern "C" int tflite_ctx_get_arena_profile(tflite_ctx_t *ctxp, tflm_arena_profile_t *out) {
    tflite_ctx *ctx = (tflite_ctx *)ctxp;
    if (!ctx || !out) return -1;
    *out = ctx->profile;
    return 0;
}

extern "C" int tflite_ctx_get_io_dtypes(tflite_ctx_t *ctxp, int *in_dtype, int *out_dtype) {
    tflite_ctx *ctx = (tflite_ctx *)ctxp;
    if (!ctx || !ctx->interpreter) return -1;
    TfLiteTensor *in_t = ctx->interpreter->input(0);
    TfLiteTensor *out_t = ctx->interpreter->output(0);
    if (!in_t || !out_t) return -1;
    if (in_dtype) *in_dtype = (int)in_t->type;
    if (out_dtype) *out_dtype = (int)out_t->type;
    return 0;
}

extern "C" int tflite_ctx_invoke(tflite_ctx_t *ctxp,
                                  const float *input, size_t input_len,
                                  float *output, size_t output_len) {
    tflite_ctx *ctx = (tflite_ctx *)ctxp;
    if (!ctx || !ctx->interpreter) return -1;

    TfLiteTensor *in_tensor = ctx->interpreter->input(0);
    TfLiteTensor *out_tensor = ctx->interpreter->output(0);

    int first_call = (s_invoke_count == 0);
    s_invoke_count++;

    if (first_call) {
        TFLM_LOGI("invoke#1: in_type=%d out_type=%d in_len=%u out_len=%u",
                   (int)in_tensor->type, (int)out_tensor->type,
                   (unsigned)input_len, (unsigned)output_len);
        TFLM_LOGI("invoke#1: input[0..3] = %.6f %.6f %.6f %.6f",
                   (double)input[0], (double)input[1],
                   (double)input[2], (double)input[3]);
    }

    /* Handle float32 or int8 input */
    if (in_tensor->type == kTfLiteFloat32) {
        memcpy(in_tensor->data.f, input, input_len * sizeof(float));
    } else if (in_tensor->type == kTfLiteInt8) {
        float scale = in_tensor->params.scale;
        int32_t zp = in_tensor->params.zero_point;
        int8_t *dst = in_tensor->data.int8;
        for (size_t i = 0; i < input_len; i++) {
            int32_t q = (int32_t)roundf(input[i] / scale) + zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst[i] = (int8_t)q;
        }
    } else {
        return -4;
    }

    if (first_call && in_tensor->type == kTfLiteInt8) {
        TFLM_LOGI("invoke#1: quant scale=%.8f zp=%d  q[0..3]=%d %d %d %d",
                   (double)in_tensor->params.scale,
                   (int)in_tensor->params.zero_point,
                   (int)in_tensor->data.int8[0], (int)in_tensor->data.int8[1],
                   (int)in_tensor->data.int8[2], (int)in_tensor->data.int8[3]);
    }

    if (ctx->interpreter->Invoke() != kTfLiteOk) return -2;

    /* Dequantize output if needed */
    if (out_tensor->type == kTfLiteFloat32) {
        size_t nbytes = output_len * sizeof(float);
        if (out_tensor->bytes < nbytes) return -3;
        memcpy(output, out_tensor->data.f, nbytes);
    } else if (out_tensor->type == kTfLiteInt8) {
        float scale = out_tensor->params.scale;
        int32_t zp = out_tensor->params.zero_point;
        const int8_t *src = out_tensor->data.int8;
        for (size_t i = 0; i < output_len; i++) {
            output[i] = ((float)src[i] - (float)zp) * scale;
        }
    } else if (out_tensor->type == kTfLiteUInt8) {
        float scale = out_tensor->params.scale;
        int32_t zp = out_tensor->params.zero_point;
        const uint8_t *src = out_tensor->data.uint8;
        for (size_t i = 0; i < output_len; i++) {
            output[i] = ((float)src[i] - (float)zp) * scale;
        }
    } else {
        return -3;
    }

    if (first_call) {
        TFLM_LOGI("invoke#1: out[0..3] = %.6f %.6f %.6f %.6f",
                   (double)output[0], (double)output[1],
                   (double)output[2], (double)output[3]);
    }

    return 0;
}

extern "C" void tflite_ctx_destroy(tflite_ctx_t *ctxp) {
    tflite_ctx *ctx = (tflite_ctx *)ctxp;
    if (!ctx) return;
    if (ctx->interpreter) delete ctx->interpreter;
    if (ctx->arena_raw) {
#ifdef ESP_PLATFORM
        heap_caps_free(ctx->arena_raw);
#else
        free(ctx->arena_raw);
#endif
    }
    free(ctx);
}
