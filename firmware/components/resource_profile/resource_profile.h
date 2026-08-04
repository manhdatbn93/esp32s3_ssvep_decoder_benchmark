/*
 * Heap / stack snapshot helpers for replay (and optional smoke) sessions.
 *
 * All "free heap" values are instantaneous samples from ESP-IDF at the call
 * site — not guaranteed peaks unless sampled after every allocation path.
 *
 * When CONFIG_REPLAY_RESOURCE_PROFILE is 0 / unset, implementations are no-ops
 * (replay firmware Kconfig enables this; other targets may omit the symbol).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t total_free;          /* esp_get_free_heap_size() */
    size_t internal_free;       /* MALLOC_CAP_INTERNAL | 8BIT */
    size_t spiram_free;         /* MALLOC_CAP_SPIRAM (0 if no PSRAM) */
    size_t internal_largest;    /* largest contiguous internal free block */
    size_t internal_min_ever;   /* heap_caps_get_minimum_free_size(INTERNAL) — since boot */
    size_t spiram_min_ever;     /* since boot, SPIRAM */
} resource_profile_heap_t;

void resource_profile_heap_read(resource_profile_heap_t *out);

/*
 * Log one line: RES_PROFILE phase=... key=value ...
 * phase: pre_backend_init | post_backend_init | post_infer | session_summary
 */
void resource_profile_log_line(const char *tag, const char *phase,
                               const char *backend_name,
                               const resource_profile_heap_t *h);

/* Call once after cleanup, immediately before backend init(). */
void resource_profile_session_pre_init(const char *tag, const char *backend_name);

/* Call once after backend init() returns success. */
void resource_profile_session_post_init(const char *tag, const char *backend_name);

/*
 * Call after each successful forward (or periodically) to track session minimum
 * free internal / SPIRAM seen on this task's timeline.
 */
void resource_profile_session_sample_after_work(void);

/* At session end: logs summary + stack high-water (words) for current task. */
void resource_profile_session_end(const char *tag, const char *backend_name,
                                  uint32_t stack_high_water_words);

/* If backend init fails after pre_backend_init, call this (no summary line). */
void resource_profile_session_cancel(void);

#ifdef __cplusplus
}
#endif
