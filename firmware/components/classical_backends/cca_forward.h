/*
 * CCA forward for 12JFPM ESP32-S3 smoke test.
 *
 * Uses precomputed flash LUT (no runtime sin/cos).
 * Precomputes reference-side centered refs + inv(Cyy) at init.
 */
#ifndef CCA_FORWARD_H
#define CCA_FORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

int cca_forward_init(void);
int cca_forward_run(const float *input_tc, float *scores_out);
void cca_forward_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CCA_FORWARD_H */
