/*
 * MSI forward for 12JFPM (ESP32-S3 + host).
 *
 * Replicates src/benchmarks/msi_12jfpm.py → _synchronization_index().
 *
 * Contract per trial:
 *   1. Input [T=512, C=8] raw float32 EEG
 *   2. Transpose to X [C=8, T=512], z-score normalize each channel
 *   3. For each of 12 target frequencies:
 *      a. Load Y [R=6, T=512] from flash LUT, z-score normalize each row
 *      b. Compute correlation matrices C11, C22, C12
 *      c. Compute C11^{-1/2}, C22^{-1/2} via eigendecomposition
 *      d. Form R = U @ C @ U^T where U = blkdiag(C11^{-1/2}, C22^{-1/2})
 *      e. Eigenvalues of R → normalized entropy → synchronization index
 *   4. Output: 12 float scores in [0,1]
 *
 * Numerical engine: Jacobi eigendecomposition for symmetric matrices
 * (sizes 6, 8, 14). Converges in ~5–10 sweeps for well-conditioned
 * correlation matrices.
 */
#include "msi_forward.h"

#include <math.h>
#include <string.h>

#include "msi_config_generated.h"
#include "msi_ref_lut_generated.h"

#define MSI_N  (MSI_C + MSI_R)   /* 14: size of combined R matrix */
#define JACOBI_MAX_N 16

/*
 * Jacobi workspaces in .bss (not stack). Large stack frames here previously
 * risked overrunning the main task stack when boot smoke runs MSI before Wi-Fi
 * (heap corruption surfaced on the next malloc inside esp_wifi_init).
 * MSI is single-threaded per call — no reentrancy requirement.
 */
static double s_jacobi_a[JACOBI_MAX_N][JACOBI_MAX_N];
static double s_jacobi_v[JACOBI_MAX_N][JACOBI_MAX_N];
static double s_msi_eig_tmp8[MSI_C * MSI_C];
static double s_msi_eig_tmp6[MSI_R * MSI_R];

/* ── Jacobi symmetric eigendecomposition (up to 16×16) ─────────── */

static void jacobi_eig(double *A, double *eigvals, double *V, int n)
{
    if (n > JACOBI_MAX_N || n <= 0) {
        return;
    }

    double (*a)[JACOBI_MAX_N] = s_jacobi_a;
    double (*v)[JACOBI_MAX_N] = s_jacobi_v;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = A[i * n + j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            v[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += a[i][j] * a[i][j];
        if (off < 1e-30) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                if (fabs(a[p][q]) < 1e-15) continue;

                double diff = a[q][q] - a[p][p];
                double t;
                if (fabs(diff) < 1e-15) {
                    t = (a[p][q] > 0) ? 1.0 : -1.0;
                } else {
                    double tau = diff / (2.0 * a[p][q]);
                    t = 1.0 / (fabs(tau) + sqrt(1.0 + tau * tau));
                    if (tau < 0) t = -t;
                }

                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                double app = a[p][p] - t * a[p][q];
                double aqq = a[q][q] + t * a[p][q];
                a[p][p] = app;
                a[q][q] = aqq;
                a[p][q] = a[q][p] = 0.0;

                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    double arp = a[r][p], arq = a[r][q];
                    a[r][p] = a[p][r] = c * arp - s * arq;
                    a[r][q] = a[q][r] = s * arp + c * arq;
                }

                for (int r = 0; r < n; r++) {
                    double vrp = v[r][p], vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) eigvals[i] = a[i][i];
    if (V)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                V[i * n + j] = v[i][j];
}

/* ── Compute M^{-1/2} = Q diag(d^{-0.5}) Q^T for symmetric M ──── */

static void mat_inv_sqrt(const double *eigvals, const double *V, int n, double *out)
{
    double d_inv[16];
    for (int i = 0; i < n; i++) {
        double d = eigvals[i];
        d_inv[i] = (d < 1e-12) ? 1.0 / sqrt(1e-12) : 1.0 / sqrt(d);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += V[i * n + k] * d_inv[k] * V[j * n + k];
            out[i * n + j] = out[j * n + i] = s;
        }
}

/* ── Public API ────────────────────────────────────────────────── */

int msi_forward_init(void)  { return 0; }
void msi_forward_cleanup(void) { }

int msi_forward_run(const float *input_tc, float *scores_out)
{
    static double X[MSI_C][MSI_T];
    static double Y[MSI_R][MSI_T];

    for (int c = 0; c < MSI_C; c++) {
        double sum = 0.0;
        for (int t = 0; t < MSI_T; t++) {
            X[c][t] = (double)input_tc[t * MSI_C + c];
            sum += X[c][t];
        }
        double mean = sum / MSI_T;
        double var = 0.0;
        for (int t = 0; t < MSI_T; t++) {
            X[c][t] -= mean;
            var += X[c][t] * X[c][t];
        }
        double sd = sqrt(var / MSI_T);
        if (sd < 1e-12) sd = 1.0;
        double inv_sd = 1.0 / sd;
        for (int t = 0; t < MSI_T; t++) X[c][t] *= inv_sd;
    }

    /* C11 = X X^T / T  (8×8 symmetric) */
    static double C11_flat[MSI_C * MSI_C];
    for (int i = 0; i < MSI_C; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int t = 0; t < MSI_T; t++) s += X[i][t] * X[j][t];
            C11_flat[i * MSI_C + j] = C11_flat[j * MSI_C + i] = s / MSI_T;
        }

    /* Eigendecomposition of C11 → C11^{-1/2} */
    static double ev_c11[MSI_C], V_c11[MSI_C * MSI_C];
    {
        memcpy(s_msi_eig_tmp8, C11_flat, sizeof(s_msi_eig_tmp8));
        jacobi_eig(s_msi_eig_tmp8, ev_c11, V_c11, MSI_C);
    }
    static double C11_isqrt[MSI_C * MSI_C];
    mat_inv_sqrt(ev_c11, V_c11, MSI_C, C11_isqrt);

    /* Per-class scoring */
    for (int cls = 0; cls < MSI_NUM_CLASSES; cls++) {

        /* Load + z-score normalize Y from flash LUT */
        for (int r = 0; r < MSI_R; r++) {
            double sum = 0.0;
            for (int t = 0; t < MSI_T; t++) {
                Y[r][t] = (double)MSI_REF_LUT[cls][r][t];
                sum += Y[r][t];
            }
            double mean = sum / MSI_T;
            double var = 0.0;
            for (int t = 0; t < MSI_T; t++) {
                Y[r][t] -= mean;
                var += Y[r][t] * Y[r][t];
            }
            double sd = sqrt(var / MSI_T);
            if (sd < 1e-12) sd = 1.0;
            double inv_sd = 1.0 / sd;
            for (int t = 0; t < MSI_T; t++) Y[r][t] *= inv_sd;
        }

        /* C22 = Y Y^T / T  (6×6 symmetric) */
        static double C22_flat[MSI_R * MSI_R];
        for (int i = 0; i < MSI_R; i++)
            for (int j = 0; j <= i; j++) {
                double s = 0.0;
                for (int t = 0; t < MSI_T; t++) s += Y[i][t] * Y[j][t];
                C22_flat[i * MSI_R + j] = C22_flat[j * MSI_R + i] = s / MSI_T;
            }

        /* Eigendecomposition of C22 → C22^{-1/2} */
        static double ev_c22[MSI_R], V_c22[MSI_R * MSI_R];
        {
            memcpy(s_msi_eig_tmp6, C22_flat, sizeof(s_msi_eig_tmp6));
            jacobi_eig(s_msi_eig_tmp6, ev_c22, V_c22, MSI_R);
        }
        static double C22_isqrt[MSI_R * MSI_R];
        mat_inv_sqrt(ev_c22, V_c22, MSI_R, C22_isqrt);

        /* C12 = X Y^T / T  (8×6) */
        static double C12[MSI_C * MSI_R];
        for (int i = 0; i < MSI_C; i++)
            for (int j = 0; j < MSI_R; j++) {
                double s = 0.0;
                for (int t = 0; t < MSI_T; t++) s += X[i][t] * Y[j][t];
                C12[i * MSI_R + j] = s / MSI_T;
            }

        /* Build C_full [14×14] and U [14×14] (block-diagonal) */
        static double C_full[MSI_N * MSI_N];
        static double U[MSI_N * MSI_N];
        memset(C_full, 0, sizeof(C_full));
        memset(U, 0, sizeof(U));

        for (int i = 0; i < MSI_C; i++)
            for (int j = 0; j < MSI_C; j++) {
                C_full[i * MSI_N + j] = C11_flat[i * MSI_C + j];
                U[i * MSI_N + j] = C11_isqrt[i * MSI_C + j];
            }
        for (int i = 0; i < MSI_R; i++)
            for (int j = 0; j < MSI_R; j++) {
                C_full[(MSI_C + i) * MSI_N + (MSI_C + j)] = C22_flat[i * MSI_R + j];
                U[(MSI_C + i) * MSI_N + (MSI_C + j)] = C22_isqrt[i * MSI_R + j];
            }
        for (int i = 0; i < MSI_C; i++)
            for (int j = 0; j < MSI_R; j++) {
                C_full[i * MSI_N + (MSI_C + j)] = C12[i * MSI_R + j];
                C_full[(MSI_C + j) * MSI_N + i] = C12[i * MSI_R + j];
            }

        /* R = U @ C_full @ U^T */
        static double tmp_mat[MSI_N * MSI_N];
        static double R_mat[MSI_N * MSI_N];

        for (int i = 0; i < MSI_N; i++)
            for (int j = 0; j < MSI_N; j++) {
                double s = 0.0;
                for (int k = 0; k < MSI_N; k++)
                    s += U[i * MSI_N + k] * C_full[k * MSI_N + j];
                tmp_mat[i * MSI_N + j] = s;
            }
        for (int i = 0; i < MSI_N; i++)
            for (int j = 0; j < MSI_N; j++) {
                double s = 0.0;
                for (int k = 0; k < MSI_N; k++)
                    s += tmp_mat[i * MSI_N + k] * U[j * MSI_N + k];
                R_mat[i * MSI_N + j] = s;
            }

        /* Eigenvalues of R (14×14 symmetric) */
        static double ev_r[MSI_N];
        jacobi_eig(R_mat, ev_r, NULL, MSI_N);

        /* Clamp and normalize → entropy → synchronization index */
        double eig_sum = 0.0;
        for (int i = 0; i < MSI_N; i++) {
            if (ev_r[i] < 1e-20) ev_r[i] = 1e-20;
            eig_sum += ev_r[i];
        }
        double entropy = 0.0;
        for (int i = 0; i < MSI_N; i++) {
            double e = ev_r[i] / eig_sum;
            entropy += e * log(e);
        }
        double S = 1.0 + entropy / log((double)MSI_N);

        scores_out[cls] = (float)S;
    }

    return 0;
}
