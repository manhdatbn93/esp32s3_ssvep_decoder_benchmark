/*
 * CCNN FFT preprocessing: raw EEG → spectrum features via Goertzel DFT.
 *
 * The CCNN model was trained with FFT preprocessing outside the network
 * (src/models/ccnn.py → _complex_spectrum_features).  The TFLite model
 * is the *post-FFT* network, so this module replicates the host-side
 * FFT preprocessing on the ESP32-S3.
 *
 * Why Goertzel instead of esp-dsp FFT:
 *   NFFT=874 is not a power of 2.  esp-dsp's dsps_fft2r_fc32 requires
 *   power-of-2 lengths.  Changing NFFT would alter frequency resolution
 *   and bin indices, breaking model parity.  The Goertzel algorithm
 *   computes exact DFT bins for arbitrary N, processing only the 110
 *   bins we need (3–35 Hz range).
 *
 * Frequency-bin mapping (derived from src/models/ccnn.py):
 *   resolution = fs / NFFT = 256.0 / 874 ≈ 0.29291 Hz/bin
 *   bin  10 →  2.93 Hz  (first bin, round(3.0 / 0.2930) = 10)
 *   bin 119 → 34.87 Hz  (last bin,  round(35.0 / 0.2930) = 119)
 *   110 bins × {real, imag} = 220 features per channel
 */
#ifndef CCNN_PREPROCESS_H
#define CCNN_PREPROCESS_H

#define CCNN_NFFT          874
#define CCNN_INPUT_T       512
#define CCNN_INPUT_C       8
#define CCNN_BIN_LO        10
#define CCNN_BIN_HI        120    /* exclusive: bins [10, 120) */
#define CCNN_N_BINS        110    /* BIN_HI - BIN_LO */
#define CCNN_N_FEATURES    220    /* 2 * N_BINS (real || imag) */
#define CCNN_FEATURE_LEN   1760   /* INPUT_C * N_FEATURES */
#define CCNN_NORM_FACTOR   256.0f /* T / 2 = 512 / 2 */

/*
 * Precompute Goertzel twiddle factors (cos/sin/coeff for bins 10–119).
 * Call once before first use; idempotent.
 */
void ccnn_preprocess_init(void);

/*
 * Convert raw [512, 8] EEG to [8, 220] spectrum features.
 *
 * input_tc : 4096 floats, [time=512, channel=8] row-major.
 * output   : 1760 floats, [channel=8, feature=220] row-major.
 *            Per channel: 110 real bins then 110 imag bins.
 *            Matches TFLite NHWC [1, 8, 220, 1] layout.
 */
void ccnn_preprocess(const float *input_tc, float *output);

#endif /* CCNN_PREPROCESS_H */
