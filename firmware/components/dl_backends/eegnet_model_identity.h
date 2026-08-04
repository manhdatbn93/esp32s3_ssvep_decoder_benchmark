/*
 * EEGNet embedded TFLite blob identity for replay START_ACK.
 */
#ifndef EEGNET_MODEL_IDENTITY_H
#define EEGNET_MODEL_IDENTITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *data;
    size_t len;
    uint32_t crc32;
    uint8_t first16[16];
    uint8_t last16[16];
    int input_dtype;   /* TfLiteType value, e.g. 1=float32 */
    int output_dtype;
    const char *symbol;
    const char *actual_backend;
} eegnet_model_identity_t;

#ifdef SMOKE_ENABLE_EEGNET_INT8
int eegnet_i8_model_identity(eegnet_model_identity_t *out);
void eegnet_i8_log_fw_model_identity(const char *requested_backend);
#endif

#ifdef SMOKE_ENABLE_EEGNET_F32
int eegnet_f32_model_identity(eegnet_model_identity_t *out);
void eegnet_f32_log_fw_model_identity(const char *requested_backend);
#endif

#ifdef __cplusplus
}
#endif

#endif /* EEGNET_MODEL_IDENTITY_H */
