/*
 * CCA forward for 12JFPM (ESP32-S3 + host).
 *
 * All fixed references are loaded from flash (cca_ref_lut_generated.h).
 * At init, centered references and inv(Cyy) per class are precomputed.
 * Per trial: SOS filtfilt → mean-center X → Cxx/inv(Cxx) → per-class CCA.
 *
 * Ported and simplified from existing_work/embedded/c_smoke/cca_forward.c.
 */
#include "cca_forward.h"
#include "sosfilt_mcu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Generated headers — flash LUT + config */
#include "cca_config_generated.h"
#include "cca_ref_lut_generated.h"

#define CCA_LEN_EXT (CCA_T + 2 * CCA_EDGE)

/* ---- Gauss-Jordan matrix inverse (n <= 8) ---- */
static double s_aug[8][16];

static int mat_inv(int n, double *a) {
    double (*const aug)[16] = s_aug;
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
        if (piv != k) {
            for (int j = 0; j < 2 * n; j++) {
                double t = aug[k][j];
                aug[k][j] = aug[piv][j];
                aug[piv][j] = t;
            }
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

/* ---- Power iteration for max eigenvalue of 8x8 matrix ---- */
static double rho_sq_from_mat8(const double mat[8][8]) {
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

/* ---- Precomputed reference-side storage ---- */
static double *g_yc_centered = NULL;   /* [NUM_CLASSES][T][R] in double */
static double *g_inv_cyy     = NULL;   /* [NUM_CLASSES][R][R] in double */
static int     g_init_done   = 0;

int cca_forward_init(void) {
    if (g_init_done) return 0;

    const int T = CCA_T;
    const int R = CCA_R;
    const int NC = CCA_NUM_CLASSES;
    const size_t per_cls = (size_t)T * (size_t)R;
    const double n = (T > 1) ? (double)(T - 1) : 1.0;
    const double reg = (double)CCA_REG;

    g_yc_centered = (double *)malloc((size_t)NC * per_cls * sizeof(double));
    g_inv_cyy     = (double *)malloc((size_t)NC * (size_t)(R * R) * sizeof(double));
    if (!g_yc_centered || !g_inv_cyy) return -1;

    for (int cls = 0; cls < NC; cls++) {
        /* Copy flash float LUT [R][T] into double [T][R] (transposed) */
        double *yc = g_yc_centered + (size_t)cls * per_cls;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++)
                yc[t * R + r] = (double)CCA_REF_LUT[cls][r][t];

        /* Mean-center each reference row */
        double mean_r[6];
        for (int r = 0; r < R; r++) {
            double s = 0.0;
            for (int t = 0; t < T; t++) s += yc[t * R + r];
            mean_r[r] = s / (double)T;
        }
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++)
                yc[t * R + r] -= mean_r[r];

        /* Compute Cyy = Y'Y/(n-1) + reg*I */
        double cyy[6][6];
        for (int i = 0; i < R; i++)
            for (int j = 0; j < R; j++) {
                double s = 0.0;
                for (int t = 0; t < T; t++)
                    s += yc[t * R + i] * yc[t * R + j];
                cyy[i][j] = s / n + ((i == j) ? reg : 0.0);
            }

        /* Invert Cyy */
        double inv_w[36];
        memcpy(inv_w, cyy, (size_t)(R * R) * sizeof(double));
        if (mat_inv(R, inv_w) != 0) return -2;
        memcpy(g_inv_cyy + (size_t)cls * (size_t)(R * R), inv_w,
               (size_t)(R * R) * sizeof(double));
    }

    g_init_done = 1;
    return 0;
}

int cca_forward_run(const float *input_tc, float *scores_out) {
    if (!g_init_done) return -1;

    const int T = CCA_T;
    const int C = CCA_C;
    const int R = CCA_R;
    const int NC = CCA_NUM_CLASSES;
    const double n = (T > 1) ? (double)(T - 1) : 1.0;
    const double reg = (double)CCA_REG;

    /* Convert input to double [T][C] and apply SOS filtfilt per channel */
    static double xb[512][8];
    static double ext[CCA_LEN_EXT];
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            xb[t][c] = (double)input_tc[t * C + c];

    for (int c = 0; c < C; c++) {
        static double ch[512];
        for (int t = 0; t < T; t++) ch[t] = xb[t][c];
        sosfilt_odd_ext_1d(ch, T, CCA_EDGE, ext);
        sosfilt_filtfilt(CCA_SOS, CCA_SOSFILT_ZI, CCA_N_SOS, ext, CCA_LEN_EXT);
        for (int t = 0; t < T; t++) xb[t][c] = ext[CCA_EDGE + t];
    }

    /* Mean-center X */
    static double xc[512][8];
    double mean_c[8];
    for (int c = 0; c < C; c++) {
        double s = 0.0;
        for (int t = 0; t < T; t++) s += xb[t][c];
        mean_c[c] = s / (double)T;
    }
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            xc[t][c] = xb[t][c] - mean_c[c];

    /* Cxx = X'X/(n-1) + reg*I, then inv(Cxx) */
    static double cxx[8][8], inv_cxx[8][8];
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) {
            double s = 0.0;
            for (int t = 0; t < T; t++) s += xc[t][i] * xc[t][j];
            cxx[i][j] = s / n + ((i == j) ? reg : 0.0);
        }
    memcpy(inv_cxx, cxx, sizeof(cxx));
    if (mat_inv(C, &inv_cxx[0][0]) != 0) return -4;

    /* Per-class: canonical correlation */
    static double cxy[8][6], tmp86[8][6], S[8][8], mat[8][8];

    for (int cls = 0; cls < NC; cls++) {
        const double *yc = g_yc_centered + (size_t)cls * (size_t)(T * R);
        const double *inv_cyy = g_inv_cyy + (size_t)cls * (size_t)(R * R);

        /* Cxy = X'Y/(n-1) */
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

        /* tmp86 = Cxy * inv_cyy  (8x6 * 6x6) */
        memset(tmp86, 0, sizeof(tmp86));
        for (int i = 0; i < C; i++)
            for (int k = 0; k < R; k++) {
                double xik = cxy[i][k];
                for (int j = 0; j < R; j++)
                    tmp86[i][j] += xik * inv_cyy[k * R + j];
            }

        /* S = tmp86 * Cxy'  (8x6 * 6x8 → 8x8) */
        for (int i = 0; i < C; i++)
            for (int q = 0; q < C; q++) {
                double s = 0.0;
                for (int j = 0; j < R; j++)
                    s += tmp86[i][j] * cxy[q][j];
                S[i][q] = s;
            }

        /* mat = inv_cxx * S  (8x8 * 8x8) */
        memset(mat, 0, sizeof(mat));
        for (int p = 0; p < C; p++)
            for (int i = 0; i < C; i++) {
                double a = inv_cxx[p][i];
                for (int q = 0; q < C; q++)
                    mat[p][q] += a * S[i][q];
            }

        double rho_sq = rho_sq_from_mat8(mat);
        if (rho_sq < 0.0) rho_sq = 0.0;
        if (rho_sq > 1.0) rho_sq = 1.0;
        scores_out[cls] = (float)sqrt(rho_sq);
    }

    return 0;
}

void cca_forward_cleanup(void) {
    free(g_yc_centered);
    free(g_inv_cyy);
    g_yc_centered = NULL;
    g_inv_cyy = NULL;
    g_init_done = 0;
}
