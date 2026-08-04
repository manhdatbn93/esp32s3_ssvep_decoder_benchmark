/*
 * MSI (Multivariate Synchronization Index) forward for 12JFPM ESP32-S3.
 *
 * Training-free method: for each candidate frequency, computes a
 * synchronization index between z-scored multichannel EEG and
 * sin/cos reference signals via eigenvalue decomposition of a
 * normalized correlation matrix.
 *
 * Reference: Zhang et al. (2014).
 *
 * All reference signals are loaded from the flash LUT
 * (msi_ref_lut_generated.h). No filter bank — single-pass on raw data.
 */
#ifndef MSI_FORWARD_H
#define MSI_FORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

int msi_forward_init(void);
int msi_forward_run(const float *input_tc, float *scores_out);
void msi_forward_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* MSI_FORWARD_H */
