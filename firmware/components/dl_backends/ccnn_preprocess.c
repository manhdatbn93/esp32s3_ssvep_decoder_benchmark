/*
 * CCNN FFT preprocessing via Goertzel algorithm.
 *
 * Replicates the host-side _complex_spectrum_features() from
 * src/models/ccnn.py on the ESP32-S3 without requiring a power-of-2
 * FFT library.
 *
 * Contract (frozen from host training code):
 *   1. Input:  [T=512, C=8] raw EEG, float32, TC row-major
 *   2. Per channel: 874-point DFT of zero-padded 512-sample signal
 *   3. Normalize: divide complex spectrum by T/2 = 256.0
 *   4. Extract bins [10, 120) → 110 complex values per channel
 *   5. Concatenate real || imag → 220 features per channel
 *   6. Output: [C=8, 220] row-major → NHWC [1, 8, 220, 1] for TFLite
 *
 * Goertzel output formula (verified against torch.fft.fft):
 *   After processing all NFFT samples through the Goertzel recurrence
 *   s[n] = x[n] + 2·cos(ω)·s[n−1] − s[n−2],  ω = 2πk/N:
 *
 *   Re(X[k]) = cos(ω)·s1 − s2      where s1 = s[N−1], s2 = s[N−2]
 *   Im(X[k]) = sin(ω)·s1
 *
 * Performance estimate: 874 × 110 × 8 ≈ 769K iterations, ~10 ms @ 240 MHz.
 */
#include "ccnn_preprocess.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float s_goertzel_coeff[CCNN_N_BINS];  /* 2·cos(ω_k) */
static float s_cos_omega[CCNN_N_BINS];
static float s_sin_omega[CCNN_N_BINS];
static int   s_initialized = 0;

void ccnn_preprocess_init(void)
{
    for (int i = 0; i < CCNN_N_BINS; i++) {
        int k = CCNN_BIN_LO + i;
        float omega = 2.0f * (float)M_PI * (float)k / (float)CCNN_NFFT;
        s_cos_omega[i] = cosf(omega);
        s_sin_omega[i] = sinf(omega);
        s_goertzel_coeff[i] = 2.0f * s_cos_omega[i];
    }
    s_initialized = 1;
}

void ccnn_preprocess(const float *input_tc, float *output)
{
    if (!s_initialized)
        ccnn_preprocess_init();

    for (int c = 0; c < CCNN_INPUT_C; c++) {
        float s1[CCNN_N_BINS];
        float s2[CCNN_N_BINS];

        for (int bi = 0; bi < CCNN_N_BINS; bi++) {
            s1[bi] = 0.0f;
            s2[bi] = 0.0f;
        }

        /* Goertzel over the 512 non-zero samples */
        for (int n = 0; n < CCNN_INPUT_T; n++) {
            float x = input_tc[n * CCNN_INPUT_C + c];
            for (int bi = 0; bi < CCNN_N_BINS; bi++) {
                float s0 = x + s_goertzel_coeff[bi] * s1[bi] - s2[bi];
                s2[bi] = s1[bi];
                s1[bi] = s0;
            }
        }

        /* Continue over the zero-padded region (874 − 512 = 362 zeros) */
        for (int n = CCNN_INPUT_T; n < CCNN_NFFT; n++) {
            for (int bi = 0; bi < CCNN_N_BINS; bi++) {
                float s0 = s_goertzel_coeff[bi] * s1[bi] - s2[bi];
                s2[bi] = s1[bi];
                s1[bi] = s0;
            }
        }

        /*
         * Extract real and imaginary parts, normalize by T/2.
         * Layout: output[c*220 + 0..109]   = real bins 10..119
         *         output[c*220 + 110..219]  = imag bins 10..119
         */
        float *out_real = output + c * CCNN_N_FEATURES;
        float *out_imag = out_real + CCNN_N_BINS;
        for (int bi = 0; bi < CCNN_N_BINS; bi++) {
            out_real[bi] = (s_cos_omega[bi] * s1[bi] - s2[bi]) / CCNN_NORM_FACTOR;
            out_imag[bi] = (s_sin_omega[bi] * s1[bi]) / CCNN_NORM_FACTOR;
        }
    }
}
