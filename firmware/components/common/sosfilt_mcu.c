/*
 * SOS filtfilt implementation for MCU classical backends.
 * Direct-form II transposed, scipy-compatible.
 */
#include "sosfilt_mcu.h"
#include <math.h>

void sosfilt_reverse_d(double *a, int n) {
    for (int i = 0; i < n / 2; i++) {
        double t = a[i];
        a[i] = a[n - 1 - i];
        a[n - 1 - i] = t;
    }
}

void sosfilt_odd_ext_1d(const double *x, int len, int n_pad, double *ext) {
    for (int i = 0; i < n_pad; i++) {
        ext[i] = 2.0 * x[0] - x[n_pad - i];
    }
    for (int i = 0; i < len; i++) {
        ext[n_pad + i] = x[i];
    }
    for (int i = 0; i < n_pad; i++) {
        ext[n_pad + len + i] = 2.0 * x[len - 1] - x[len - 2 - i];
    }
}

static void sosfilt_pass_loop(const double sos[][6], const double zi_init[][2],
                               int n_sos, double *ext, int len_ext, double scale_zi) {
    double zi[8][2];
    if (n_sos > 8) return;
    for (int s = 0; s < n_sos; s++) {
        zi[s][0] = zi_init[s][0] * scale_zi;
        zi[s][1] = zi_init[s][1] * scale_zi;
    }
    for (int n = 0; n < len_ext; n++) {
        double x_cur = ext[n];
        for (int s = 0; s < n_sos; s++) {
            double b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2];
            double a1 = sos[s][4], a2 = sos[s][5];
            double x_new = b0 * x_cur + zi[s][0];
            zi[s][0] = b1 * x_cur - a1 * x_new + zi[s][1];
            zi[s][1] = b2 * x_cur - a2 * x_new;
            x_cur = x_new;
        }
        ext[n] = x_cur;
    }
}

void sosfilt_pass_inplace(const double sos[][6], const double zi_init[][2],
                          int n_sos, double *ext, int len_ext, double scale_zi) {
    sosfilt_pass_loop(sos, zi_init, n_sos, ext, len_ext, scale_zi);
}

void sosfilt_filtfilt(const double sos[][6], const double zi_init[][2],
                      int n_sos, double *ext, int len_ext) {
    double x0 = ext[0];
    sosfilt_pass_inplace(sos, zi_init, n_sos, ext, len_ext, x0);
    double y0 = ext[len_ext - 1];
    sosfilt_reverse_d(ext, len_ext);
    sosfilt_pass_inplace(sos, zi_init, n_sos, ext, len_ext, y0);
    sosfilt_reverse_d(ext, len_ext);
}
