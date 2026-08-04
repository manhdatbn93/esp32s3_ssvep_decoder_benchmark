/*
 * Unified backend dispatch for optional boot self-checks and shared types.
 */
#ifndef SSVEP_DISPATCH_H
#define SSVEP_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#define SSVEP_T         512
#define SSVEP_C         8
#define SSVEP_N_CLASSES 12
#define SSVEP_MAX_BACKENDS 16

/* Compatibility macros used by existing *_smoke.c vector headers */
#define SMOKE_T         SSVEP_T
#define SMOKE_C         SSVEP_C
#define SMOKE_N_CLASSES SSVEP_N_CLASSES
#define SMOKE_MAX_BACKENDS SSVEP_MAX_BACKENDS

typedef struct {
    const char *name;

    int  (*init)(void);
    int  (*run)(const float *input_tc, float *scores_out);
    void (*cleanup)(void);

    int          n_vectors;
    const float *const *vectors;
    const int   *expected_argmax;
    int          has_expected_scores;
    const float *expected_scores_flat;
} ssvep_backend_t;

typedef struct {
    const char *name;
    int  pass;
    int  fail;
    int  skip;
    float worst_max_diff;
    float total_latency_ms;
} ssvep_result_t;

typedef ssvep_backend_t smoke_backend_t;
typedef ssvep_result_t smoke_result_t;

int ssvep_run_all(const ssvep_backend_t *backends, int n_backends,
                  ssvep_result_t *results_out);

#define smoke_run_all ssvep_run_all

#endif /* SSVEP_DISPATCH_H */
