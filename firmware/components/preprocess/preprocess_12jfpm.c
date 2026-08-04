/*
 * On-device preprocessing for 12JFPM.
 * Matches Python contract: channel z-score normalization.
 */
#include "preprocess_12jfpm.h"
#include <math.h>

void preprocess_zscore_inplace(float *data, int T, int C) {
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            sum += data[t * C + c];
        }
        float mean = sum / (float)T;

        float var_sum = 0.0f;
        for (int t = 0; t < T; t++) {
            float d = data[t * C + c] - mean;
            var_sum += d * d;
        }
        float std = sqrtf(var_sum / (float)T);
        if (std < 1e-8f) std = 1.0f;

        for (int t = 0; t < T; t++) {
            data[t * C + c] = (data[t * C + c] - mean) / std;
        }
    }
}
