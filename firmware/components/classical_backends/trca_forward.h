/*
 * TRCA (Task-Related Component Analysis) forward for 12JFPM ESP32-S3.
 *
 * Training-based method using fold-specific templates and spatial filters
 * exported from the Python training pipeline. Ensemble TRCA projects
 * both test and template through ALL class spatial filters, computes
 * Pearson correlation, and fuses across a 10-band Chebyshev filter bank.
 *
 * All fold-specific data is in flash headers:
 *   - trca_config_generated.h      (SOS, band coefs)
 *   - trca_templates_generated.h   (class templates [12][10][8][512])
 *   - trca_spatial_filters_generated.h (W [10][12][8])
 */
#ifndef TRCA_FORWARD_H
#define TRCA_FORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

int trca_forward_init(void);
int trca_forward_run(const float *input_tc, float *scores_out);
void trca_forward_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TRCA_FORWARD_H */
