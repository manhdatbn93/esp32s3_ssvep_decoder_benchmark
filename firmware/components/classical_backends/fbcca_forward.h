/*
 * FBCCA forward for 12JFPM ESP32-S3 smoke test.
 *
 * 10-band filter bank CCA with precomputed flash LUT references.
 * Fusion: score[k] = sum_b w_b * rho_{k,b}^2
 */
#ifndef FBCCA_FORWARD_H
#define FBCCA_FORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

int fbcca_forward_init(void);
int fbcca_forward_run(const float *input_tc, float *scores_out);
void fbcca_forward_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* FBCCA_FORWARD_H */
