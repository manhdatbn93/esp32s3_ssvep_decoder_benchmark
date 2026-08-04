/*
 * Embedded TFLite flatbuffer identity (CRC32 + byte probes).
 * CRC32 matches Python zlib.crc32 (IEEE polynomial).
 */
#ifndef MODEL_IDENTITY_H
#define MODEL_IDENTITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t model_identity_crc32(const uint8_t *data, size_t len);

void model_identity_hex16(const uint8_t *data, size_t len, int from_end,
                         char *out_hex, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_IDENTITY_H */
