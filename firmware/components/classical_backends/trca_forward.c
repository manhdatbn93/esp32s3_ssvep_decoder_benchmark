/*
 * TRCA forward for 12JFPM (ESP32-S3 + host).
 *
 * Replicates src/benchmarks/trca_12jfpm.py → TRCAModel.predict().
 *
 * Contract per trial:
 *   1. Input [T=512, C=8] raw float32 EEG
 *   2. Apply 10-band Chebyshev Type I filter bank (SOS filtfilt)
 *   3. For each band, for each class:
 *      a. Project filtered test through ensemble spatial filters
 *         (all 12 class filters concatenated → vector of T×K)
 *      b. Project stored template through the same filters
 *      c. Compute Pearson correlation between projected vectors
 *      d. Accumulate: rho[cls] += band_coef[band] * r
 *   4. Output: 12 float scores (argmax is the prediction)
 *
 * Memory: incremental correlation avoids storing the 6144-element
 * projected vectors — only running sums are maintained.
 */
#include "trca_forward.h"
#include "sosfilt_mcu.h"

#include <math.h>
#include <string.h>

#include "trca_config_generated.h"
#include "trca_templates_generated.h"
#include "trca_spatial_filters_generated.h"

#define TR_T     TRCA_T
#define TR_C     TRCA_C
#define TR_NC    TRCA_NUM_CLASSES
#define TR_NB    TRCA_N_BANDS

#define TR_EDGE_MAX   (3 * TRCA_N_SOS_MAX)
#define TR_LEN_EXT_MAX (TR_T + 2 * TR_EDGE_MAX)

/* ── Public API ─────────────────────────────────────────────── */

int trca_forward_init(void)  { return 0; }
void trca_forward_cleanup(void) { }

int trca_forward_run(const float *input_tc, float *scores_out)
{
    static double xb[TR_T][TR_C];
    static double filtered[TR_C][TR_T];
    static double ext[TR_LEN_EXT_MAX];
    static double ch[TR_T];

    /* Convert to double [T][C] */
    for (int t = 0; t < TR_T; t++)
        for (int c = 0; c < TR_C; c++)
            xb[t][c] = (double)input_tc[t * TR_C + c];

    /* Zero accumulated scores */
    double rho[TR_NC];
    for (int k = 0; k < TR_NC; k++) rho[k] = 0.0;

    for (int band = 0; band < TR_NB; band++) {
        int n_sos = TRCA_N_SOS_PER_BAND[band];
        int edge  = 3 * n_sos;
        int len_ext = TR_T + 2 * edge;

        /* Filter each channel through this band's SOS */
        for (int c = 0; c < TR_C; c++) {
            for (int t = 0; t < TR_T; t++) ch[t] = xb[t][c];
            sosfilt_odd_ext_1d(ch, TR_T, edge, ext);
            sosfilt_filtfilt(TRCA_SOS[band], TRCA_SOSFILT_ZI[band],
                             n_sos, ext, len_ext);
            for (int t = 0; t < TR_T; t++)
                filtered[c][t] = ext[edge + t];
        }

        double w_band = TRCA_BAND_COEFS[band];

        for (int cls = 0; cls < TR_NC; cls++) {
            /*
             * Ensemble projection + Pearson correlation.
             *
             * proj_test[t][k]     = sum_c filtered[c][t]            * W[band][k][c]
             * proj_template[t][k] = sum_c TRCA_TEMPLATES[cls][band][c][t] * W[band][k][c]
             *
             * We compute the Pearson r between the flattened
             * (T * n_classes)-length vectors without storing them.
             */
            double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
            double sum_x2 = 0.0, sum_y2 = 0.0;

            for (int t = 0; t < TR_T; t++) {
                for (int k = 0; k < TR_NC; k++) {
                    double px = 0.0, py = 0.0;
                    for (int c = 0; c < TR_C; c++) {
                        double w = (double)TRCA_SPATIAL_W[band][k][c];
                        px += filtered[c][t] * w;
                        py += (double)TRCA_TEMPLATES[cls][band][c][t] * w;
                    }
                    sum_xy += px * py;
                    sum_x  += px;  sum_y  += py;
                    sum_x2 += px * px;
                    sum_y2 += py * py;
                }
            }

            double n = (double)(TR_T * TR_NC);
            double mean_x = sum_x / n;
            double mean_y = sum_y / n;
            double var_x  = sum_x2 / n - mean_x * mean_x;
            double var_y  = sum_y2 / n - mean_y * mean_y;
            double cov_xy = sum_xy / n - mean_x * mean_y;
            double denom  = sqrt(var_x * var_y);
            double r = (denom < 1e-30) ? 0.0 : cov_xy / denom;

            rho[cls] += w_band * r;
        }
    }

    for (int k = 0; k < TR_NC; k++)
        scores_out[k] = (float)rho[k];

    return 0;
}
