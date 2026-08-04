/*
 * FBCCA forward for 12JFPM (ESP32-S3 + host).
 *
 * 10-band Chebyshev Type I filter bank, each followed by CCA per class.
 * score[k] = sum_b w_b * rho_{k,b}^2 (Pan-style fusion).
 *
 * All filter coefficients, band weights, and reference LUTs are from flash.
 * Centered refs + inv(Cyy) are precomputed once at init from the flash LUT.
 */
#include "fbcca_forward.h"
#include "sosfilt_mcu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fbcca_config_generated.h"
#include "fbcca_ref_lut_generated.h"

/* Reuse CCA-compatible dimensions (verified at compile time) */
#define FB_T    FBCCA_T
#define FB_C    FBCCA_C
#define FB_R    FBCCA_R
#define FB_NC   FBCCA_NUM_CLASSES
#define FB_NB   FBCCA_N_BANDS

/* Max edge = 3 * max_n_sos */
#define FB_EDGE_MAX (3 * FBCCA_N_SOS_MAX)
#define FB_LEN_EXT_MAX (FB_T + 2 * FB_EDGE_MAX)

/* ---- Gauss-Jordan (same as CCA, n<=8) ---- */
static double s_fb_aug[8][16];

static int fb_mat_inv(int n, double *a) {
    double (*const aug)[16] = s_fb_aug;
    if (n > 8) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = a[i * n + j];
        for (int j = 0; j < n; j++) aug[i][n + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int k = 0; k < n; k++) {
        int piv = k;
        double maxv = fabs(aug[k][k]);
        for (int i = k + 1; i < n; i++) {
            double v = fabs(aug[i][k]);
            if (v > maxv) { maxv = v; piv = i; }
        }
        if (maxv < 1e-18) return -1;
        if (piv != k)
            for (int j = 0; j < 2 * n; j++) {
                double t = aug[k][j]; aug[k][j] = aug[piv][j]; aug[piv][j] = t;
            }
        double div = aug[k][k];
        for (int j = 0; j < 2 * n; j++) aug[k][j] /= div;
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            double f = aug[i][k];
            for (int j = 0; j < 2 * n; j++) aug[i][j] -= f * aug[k][j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i * n + j] = aug[i][n + j];
    return 0;
}

/* ---- Power iteration for max eigenvalue ---- */
static double fb_rho_sq_from_mat8(const double mat[8][8]) {
    double v[8];
    for (int i = 0; i < 8; i++) v[i] = 1.0;
    for (int it = 0; it < 220; it++) {
        double w[8] = {0};
        for (int i = 0; i < 8; i++) {
            double wi = 0.0;
            for (int j = 0; j < 8; j++) wi += mat[i][j] * v[j];
            w[i] = wi;
        }
        double norm = 0.0;
        for (int i = 0; i < 8; i++) norm += w[i] * w[i];
        norm = sqrt(norm);
        if (norm < 1e-30) break;
        double inv_norm = 1.0 / norm;
        for (int i = 0; i < 8; i++) v[i] = w[i] * inv_norm;
    }
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 8; i++) {
        double mv = 0.0;
        for (int j = 0; j < 8; j++) mv += mat[i][j] * v[j];
        num += v[i] * mv;
        den += v[i] * v[i];
    }
    return (den < 1e-30) ? 0.0 : (num / den);
}

/* ---- Precomputed reference-side ---- */
static double *g_fb_yc_centered = NULL;   /* [NC][T][R] */
static double *g_fb_inv_cyy     = NULL;   /* [NC][R][R] */
static int     g_fb_init_done   = 0;

int fbcca_forward_init(void) {
    if (g_fb_init_done) return 0;

    const size_t per_cls = (size_t)FB_T * (size_t)FB_R;
    const double n = (FB_T > 1) ? (double)(FB_T - 1) : 1.0;
    const double reg = (double)FBCCA_REG;

    g_fb_yc_centered = (double *)malloc((size_t)FB_NC * per_cls * sizeof(double));
    g_fb_inv_cyy     = (double *)malloc((size_t)FB_NC * (size_t)(FB_R * FB_R) * sizeof(double));
    if (!g_fb_yc_centered || !g_fb_inv_cyy) return -1;

    for (int cls = 0; cls < FB_NC; cls++) {
        double *yc = g_fb_yc_centered + (size_t)cls * per_cls;
        /* Flash LUT is [class][row][sample]: FBCCA_REF_LUT[cls][r][t] */
        for (int t = 0; t < FB_T; t++)
            for (int r = 0; r < FB_R; r++)
                yc[t * FB_R + r] = (double)FBCCA_REF_LUT[cls][r][t];

        double mean_r[6];
        for (int r = 0; r < FB_R; r++) {
            double s = 0.0;
            for (int t = 0; t < FB_T; t++) s += yc[t * FB_R + r];
            mean_r[r] = s / (double)FB_T;
        }
        for (int t = 0; t < FB_T; t++)
            for (int r = 0; r < FB_R; r++)
                yc[t * FB_R + r] -= mean_r[r];

        double cyy[6][6];
        for (int i = 0; i < FB_R; i++)
            for (int j = 0; j < FB_R; j++) {
                double s = 0.0;
                for (int t = 0; t < FB_T; t++)
                    s += yc[t * FB_R + i] * yc[t * FB_R + j];
                cyy[i][j] = s / n + ((i == j) ? reg : 0.0);
            }

        double inv_w[36];
        memcpy(inv_w, cyy, sizeof(double) * FB_R * FB_R);
        if (fb_mat_inv(FB_R, inv_w) != 0) return -2;
        memcpy(g_fb_inv_cyy + (size_t)cls * (size_t)(FB_R * FB_R), inv_w,
               sizeof(double) * FB_R * FB_R);
    }

    g_fb_init_done = 1;
    return 0;
}

static double cca_one_class(const double xc[][8], int cls) {
    const int T = FB_T, C = FB_C, R = FB_R;
    const double n = (T > 1) ? (double)(T - 1) : 1.0;
    const double reg = (double)FBCCA_REG;

    const double *yc = g_fb_yc_centered + (size_t)cls * (size_t)(T * R);
    const double *inv_cyy = g_fb_inv_cyy + (size_t)cls * (size_t)(R * R);

    /* Cxx + inv(Cxx) */
    static double cxx[8][8], inv_cxx[8][8];
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) {
            double s = 0.0;
            for (int t = 0; t < T; t++) s += xc[t][i] * xc[t][j];
            cxx[i][j] = s / n + ((i == j) ? reg : 0.0);
        }
    memcpy(inv_cxx, cxx, sizeof(cxx));
    if (fb_mat_inv(C, &inv_cxx[0][0]) != 0) return 0.0;

    /* Cxy */
    static double cxy[8][6];
    memset(cxy, 0, sizeof(cxy));
    for (int t = 0; t < T; t++) {
        const double *xrow = xc[t];
        const double *yrow = yc + t * R;
        for (int i = 0; i < C; i++) {
            double xi = xrow[i];
            for (int j = 0; j < R; j++)
                cxy[i][j] += xi * yrow[j];
        }
    }
    {
        double inv_n = 1.0 / n;
        for (int i = 0; i < C; i++)
            for (int j = 0; j < R; j++)
                cxy[i][j] *= inv_n;
    }

    /* tmp = Cxy * inv_cyy; S = tmp * Cxy'; mat = inv_cxx * S */
    static double tmp86[8][6], S[8][8], mat[8][8];
    memset(tmp86, 0, sizeof(tmp86));
    for (int i = 0; i < C; i++)
        for (int k = 0; k < R; k++) {
            double xik = cxy[i][k];
            for (int j = 0; j < R; j++)
                tmp86[i][j] += xik * inv_cyy[k * R + j];
        }
    for (int i = 0; i < C; i++)
        for (int q = 0; q < C; q++) {
            double s = 0.0;
            for (int j = 0; j < R; j++) s += tmp86[i][j] * cxy[q][j];
            S[i][q] = s;
        }
    memset(mat, 0, sizeof(mat));
    for (int p = 0; p < C; p++)
        for (int i = 0; i < C; i++) {
            double a = inv_cxx[p][i];
            for (int q = 0; q < C; q++)
                mat[p][q] += a * S[i][q];
        }

    double rho_sq = fb_rho_sq_from_mat8(mat);
    if (rho_sq < 0.0) rho_sq = 0.0;
    if (rho_sq > 1.0) rho_sq = 1.0;
    return rho_sq;
}

int fbcca_forward_run(const float *input_tc, float *scores_out) {
    if (!g_fb_init_done) return -1;

    for (int k = 0; k < FB_NC; k++) scores_out[k] = 0.0f;

    static double xb[512][8];
    static double xc[512][8];
    static double ext[FB_LEN_EXT_MAX];

    for (int t = 0; t < FB_T; t++)
        for (int c = 0; c < FB_C; c++)
            xb[t][c] = (double)input_tc[t * FB_C + c];

    for (int band = 0; band < FB_NB; band++) {
        int n_sos = FBCCA_N_SOS_PER_BAND[band];
        int edge = 3 * n_sos;
        int len_ext = FB_T + 2 * edge;

        /* Filter each channel with this band's SOS */
        static double filtered[512][8];
        for (int c = 0; c < FB_C; c++) {
            static double ch[512];
            for (int t = 0; t < FB_T; t++) ch[t] = xb[t][c];
            sosfilt_odd_ext_1d(ch, FB_T, edge, ext);
            sosfilt_filtfilt(FBCCA_SOS[band], FBCCA_SOSFILT_ZI[band], n_sos, ext, len_ext);
            for (int t = 0; t < FB_T; t++) filtered[t][c] = ext[edge + t];
        }

        /* Mean-center */
        double mean_c[8];
        for (int c = 0; c < FB_C; c++) {
            double s = 0.0;
            for (int t = 0; t < FB_T; t++) s += filtered[t][c];
            mean_c[c] = s / (double)FB_T;
        }
        for (int t = 0; t < FB_T; t++)
            for (int c = 0; c < FB_C; c++)
                xc[t][c] = filtered[t][c] - mean_c[c];

        /* CCA per class, fuse with band weight */
        double w = FBCCA_BAND_WEIGHTS[band];
        for (int cls = 0; cls < FB_NC; cls++) {
            double rho_sq = cca_one_class(xc, cls);
            scores_out[cls] += (float)(w * rho_sq);
        }
    }

    return 0;
}

void fbcca_forward_cleanup(void) {
    free(g_fb_yc_centered);
    free(g_fb_inv_cyy);
    g_fb_yc_centered = NULL;
    g_fb_inv_cyy = NULL;
    g_fb_init_done = 0;
}
