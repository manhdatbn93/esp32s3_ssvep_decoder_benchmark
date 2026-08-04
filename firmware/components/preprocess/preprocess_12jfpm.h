/*
 * 12JFPM preprocessing constants for on-device inference.
 *
 * These match the frozen Python preprocessing contract:
 *   - fs=256 Hz, onset=38, window=512, channels=8, classes=12
 *   - Channel z-score normalization
 */
#ifndef PREPROCESS_12JFPM_H
#define PREPROCESS_12JFPM_H

#define PREPROC_FS_HZ 256.0f
#define PREPROC_ONSET_SAMPLE 38
#define PREPROC_T 512
#define PREPROC_C 8
#define PREPROC_N_CLASSES 12

/*
 * Apply channel-wise z-score normalization in place.
 * data: row-major [T x C] float32.
 */
void preprocess_zscore_inplace(float *data, int T, int C);

#endif /* PREPROCESS_12JFPM_H */
