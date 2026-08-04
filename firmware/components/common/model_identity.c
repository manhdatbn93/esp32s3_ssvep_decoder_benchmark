#include "model_identity.h"

#include <stdio.h>
#include <string.h>

uint32_t model_identity_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; bit++) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

void model_identity_hex16(const uint8_t *data, size_t len, int from_end,
                          char *out_hex, size_t out_cap) {
    if (!out_hex || out_cap < 33) {
        if (out_hex && out_cap > 0) out_hex[0] = '\0';
        return;
    }
    const uint8_t *p = data;
    size_t n = len >= 16 ? 16 : len;
    if (from_end && len >= 16) p = data + len - 16;
    for (size_t i = 0; i < n; i++)
        snprintf(out_hex + i * 2, out_cap - i * 2, "%02x", p[i]);
    out_hex[n * 2] = '\0';
}
