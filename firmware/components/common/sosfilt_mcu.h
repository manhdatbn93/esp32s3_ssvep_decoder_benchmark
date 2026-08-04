/*
 * SOS filtfilt implementation for MCU.
 *
 * Shared MCU SOS helper used by classical backends.
 * Direct-form II transposed, scipy-compatible odd extension + filtfilt.
 */
#ifndef SOSFILT_MCU_H
#define SOSFILT_MCU_H

void sosfilt_reverse_d(double *a, int n);
void sosfilt_odd_ext_1d(const double *x, int len, int n_pad, double *ext);
void sosfilt_pass_inplace(const double sos[][6], const double zi_init[][2], int n_sos,
                          double *ext, int len_ext, double scale_zi);
void sosfilt_filtfilt(const double sos[][6], const double zi_init[][2], int n_sos,
                      double *ext, int len_ext);

#endif /* SOSFILT_MCU_H */
