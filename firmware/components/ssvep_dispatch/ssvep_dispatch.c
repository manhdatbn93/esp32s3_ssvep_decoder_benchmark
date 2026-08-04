/*
 * Unified dispatch: feeds embedded test vectors to each backend,
 * checks argmax parity, computes max_abs_diff against expected scores,
 * and logs timing via esp_timer (or CLOCK_MONOTONIC on host).
 */
#include "ssvep_dispatch.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_log.h"
static const char *TAG = "ssvep_dispatch";
#define LOG_I(fmt, ...) ESP_LOGI(TAG, fmt, ##__VA_ARGS__)
#define LOG_W(fmt, ...) ESP_LOGW(TAG, fmt, ##__VA_ARGS__)
#define LOG_E(fmt, ...) ESP_LOGE(TAG, fmt, ##__VA_ARGS__)
#else
#include <time.h>
#define LOG_I(fmt, ...) printf("[I] " fmt "\n", ##__VA_ARGS__)
#define LOG_W(fmt, ...) printf("[W] " fmt "\n", ##__VA_ARGS__)
#define LOG_E(fmt, ...) printf("[E] " fmt "\n", ##__VA_ARGS__)
static long long get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}
#endif

static int argmax_f(const float *a, int n) {
    int best = 0;
    float best_val = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > best_val) {
            best_val = a[i];
            best = i;
        }
    }
    return best;
}

static float compute_max_abs_diff(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0.0f) d = -d;
        if (d > mx) mx = d;
    }
    return mx;
}

static int validate_backend(const ssvep_backend_t *be, int idx) {
    if (!be->name) {
        LOG_E("  backend[%d]: name is NULL — SKIP", idx);
        return -1;
    }
    if (!be->run) {
        LOG_E("  backend[%d] '%s': run is NULL — SKIP", idx, be->name);
        return -1;
    }
    if (be->n_vectors > 0 && !be->vectors) {
        LOG_E("  backend[%d] '%s': n_vectors=%d but vectors=NULL — SKIP",
              idx, be->name, be->n_vectors);
        return -1;
    }
    if (be->has_expected_scores && !be->expected_scores_flat) {
        LOG_W("  backend[%d] '%s': has_expected_scores=1 but pointer=NULL — disabling scores check",
              idx, be->name);
        return 1;
    }
    return 0;
}

int ssvep_run_all(const ssvep_backend_t *backends, int n_backends,
                  ssvep_result_t *results_out) {
    float scores[SSVEP_N_CLASSES];
    int total_clean = 0;

    LOG_I("========================================");
    LOG_I("  SSVEP DISPATCH — %d backends", n_backends);
    LOG_I("========================================");

    for (int b = 0; b < n_backends; b++) {
        const ssvep_backend_t *be = &backends[b];
        ssvep_result_t *res = &results_out[b];
        memset(res, 0, sizeof(*res));
        res->name = be->name ? be->name : "(null)";

        int vrc = validate_backend(be, b);
        if (vrc < 0) {
            res->skip = 1;
            continue;
        }
        int scores_ok = (vrc == 0) && be->has_expected_scores && be->expected_scores_flat;

        LOG_I("--- Backend: %s (%d vectors, scores_check=%d) ---",
              be->name, be->n_vectors, scores_ok);

        if (be->n_vectors == 0 || be->vectors == NULL) {
            LOG_W("  No boot vectors — SKIP");
            res->skip = 1;
            continue;
        }

        if (be->init) {
            int rc = be->init();
            if (rc != 0) {
                LOG_E("  INIT FAILED (rc=%d) — SKIP", rc);
                res->skip = 1;
                continue;
            }
        }

        for (int v = 0; v < be->n_vectors; v++) {
            const float *vec = be->vectors[v];
            if (!vec) {
                LOG_E("  vec[%d]: vector pointer is NULL — skip", v);
                res->fail++;
                continue;
            }

            memset(scores, 0, sizeof(scores));

#ifdef ESP_PLATFORM
            int64_t t0 = esp_timer_get_time();
#else
            long long t0 = get_time_us();
#endif
            int rc = be->run(vec, scores);

#ifdef ESP_PLATFORM
            int64_t elapsed_us = esp_timer_get_time() - t0;
#else
            long long elapsed_us = get_time_us() - t0;
#endif

            float latency_ms = (float)elapsed_us / 1000.0f;
            res->total_latency_ms += latency_ms;

            if (rc != 0) {
                LOG_E("  vec[%d]: RUN FAILED (rc=%d)", v, rc);
                res->fail++;
                continue;
            }

            int pred = argmax_f(scores, SSVEP_N_CLASSES);
            int expect = be->expected_argmax ? be->expected_argmax[v] : -1;
            int am_match = (expect < 0) || (pred == expect);

            float mdiff = -1.0f;
            if (scores_ok) {
                const float *ref = be->expected_scores_flat + v * SSVEP_N_CLASSES;
                mdiff = compute_max_abs_diff(scores, ref, SSVEP_N_CLASSES);
                if (mdiff > res->worst_max_diff)
                    res->worst_max_diff = mdiff;
            }

            if (am_match) {
                res->pass++;
            } else {
                res->fail++;
            }

            if (mdiff >= 0.0f) {
                LOG_I("  vec[%d]: pred=%2d exp=%2d %s  max_diff=%.6f  %.1f ms",
                      v, pred, expect, am_match ? "PASS" : "FAIL",
                      (double)mdiff, (double)latency_ms);
            } else {
                LOG_I("  vec[%d]: pred=%2d exp=%2d %s  %.1f ms",
                      v, pred, expect, am_match ? "PASS" : "FAIL",
                      (double)latency_ms);
            }
        }

        LOG_I("  Result: %d/%d passed, worst_max_diff=%.6f, total=%.1f ms",
              res->pass, res->pass + res->fail,
              (double)res->worst_max_diff, (double)res->total_latency_ms);

        if (be->cleanup) {
            be->cleanup();
        }

        if (res->fail == 0 && res->pass > 0)
            total_clean++;
    }

    LOG_I("========================================");
    LOG_I("  FINAL SUMMARY");
    LOG_I("========================================");
    for (int b = 0; b < n_backends; b++) {
        const ssvep_result_t *r = &results_out[b];
        if (r->skip) {
            LOG_W("  %-20s SKIP", r->name);
        } else if (r->fail == 0 && r->pass > 0) {
            LOG_I("  %-20s PASS (%d/%d, max_diff=%.6f, %.1f ms)",
                  r->name, r->pass, r->pass + r->fail,
                  (double)r->worst_max_diff, (double)r->total_latency_ms);
        } else {
            LOG_E("  %-20s FAIL (%d/%d)", r->name, r->pass, r->pass + r->fail);
        }
    }
    LOG_I("  %d/%d backends clean", total_clean, n_backends);
    LOG_I("========================================");

    return total_clean;
}
