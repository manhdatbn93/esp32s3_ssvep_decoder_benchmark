/*
 * Resource profiling for on-device replay sessions.
 */
#include "resource_profile.h"

#include <stdint.h>
#include <stddef.h>

#include "sdkconfig.h"

#if CONFIG_REPLAY_RESOURCE_PROFILE

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static size_t s_sess_baseline_internal = 0;
static size_t s_sess_min_internal = SIZE_MAX;
static size_t s_sess_min_spiram = SIZE_MAX;
static int    s_sess_active = 0;

void resource_profile_heap_read(resource_profile_heap_t *out)
{
    if (!out) {
        return;
    }
    out->total_free = esp_get_free_heap_size();
    out->internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    out->spiram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    out->internal_largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    out->internal_min_ever = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    out->spiram_min_ever = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
}

void resource_profile_log_line(const char *tag, const char *phase,
                               const char *backend_name,
                               const resource_profile_heap_t *h)
{
    if (!h) {
        return;
    }
    ESP_LOGI(
        tag,
        "RES_PROFILE phase=%s backend=%s total_free=%u int_free=%u spiram_free=%u "
        "int_largest=%u int_min_ever_boot=%u spiram_min_ever_boot=%u",
        phase, backend_name ? backend_name : "?",
        (unsigned)h->total_free, (unsigned)h->internal_free, (unsigned)h->spiram_free,
        (unsigned)h->internal_largest, (unsigned)h->internal_min_ever,
        (unsigned)h->spiram_min_ever);
}

void resource_profile_session_pre_init(const char *tag, const char *backend_name)
{
    resource_profile_heap_t h;
    resource_profile_heap_read(&h);
    s_sess_baseline_internal = h.internal_free;
    s_sess_min_internal = h.internal_free;
    s_sess_min_spiram = h.spiram_free;
    s_sess_active = 1;
    resource_profile_log_line(tag, "pre_backend_init", backend_name, &h);
}

void resource_profile_session_post_init(const char *tag, const char *backend_name)
{
    resource_profile_heap_t h;
    resource_profile_heap_read(&h);
    if (s_sess_active) {
        if (h.internal_free < s_sess_min_internal) {
            s_sess_min_internal = h.internal_free;
        }
        if (h.spiram_free < s_sess_min_spiram) {
            s_sess_min_spiram = h.spiram_free;
        }
    }
    resource_profile_log_line(tag, "post_backend_init", backend_name, &h);
}

void resource_profile_session_sample_after_work(void)
{
    if (!s_sess_active) {
        return;
    }
    size_t fi = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t fs = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (fi < s_sess_min_internal) {
        s_sess_min_internal = fi;
    }
    if (fs < s_sess_min_spiram) {
        s_sess_min_spiram = fs;
    }
}

void resource_profile_session_end(const char *tag, const char *backend_name,
                                  uint32_t stack_high_water_words)
{
    resource_profile_heap_t h;
    resource_profile_heap_read(&h);

    size_t est_drop = 0;
    if (s_sess_active && s_sess_baseline_internal >= s_sess_min_internal) {
        est_drop = s_sess_baseline_internal - s_sess_min_internal;
    }

    ESP_LOGI(
        tag,
        "RES_PROFILE phase=session_summary backend=%s int_free_now=%u spiram_free_now=%u "
        "sess_baseline_int=%u sess_min_int=%u sess_est_peak_drop_int=%u "
        "sess_min_spiram=%u stack_hwm_words=%u int_min_ever_boot=%u note=see_docs",
        backend_name ? backend_name : "?",
        (unsigned)h.internal_free, (unsigned)h.spiram_free,
        (unsigned)s_sess_baseline_internal, (unsigned)s_sess_min_internal,
        (unsigned)est_drop, (unsigned)s_sess_min_spiram,
        (unsigned)stack_high_water_words, (unsigned)h.internal_min_ever);

    s_sess_active = 0;
    s_sess_min_internal = SIZE_MAX;
    s_sess_min_spiram = SIZE_MAX;
    s_sess_baseline_internal = 0;
}

void resource_profile_session_cancel(void)
{
    s_sess_active = 0;
    s_sess_min_internal = SIZE_MAX;
    s_sess_min_spiram = SIZE_MAX;
    s_sess_baseline_internal = 0;
}

#else /* !CONFIG_REPLAY_RESOURCE_PROFILE or unset */

void resource_profile_heap_read(resource_profile_heap_t *out)
{
    if (out) {
        out->total_free = 0;
        out->internal_free = 0;
        out->spiram_free = 0;
        out->internal_largest = 0;
        out->internal_min_ever = 0;
        out->spiram_min_ever = 0;
    }
}

void resource_profile_log_line(const char *tag, const char *phase,
                               const char *backend_name,
                               const resource_profile_heap_t *h)
{
    (void)tag;
    (void)phase;
    (void)backend_name;
    (void)h;
}

void resource_profile_session_pre_init(const char *tag, const char *backend_name)
{
    (void)tag;
    (void)backend_name;
}

void resource_profile_session_post_init(const char *tag, const char *backend_name)
{
    (void)tag;
    (void)backend_name;
}

void resource_profile_session_sample_after_work(void) {}

void resource_profile_session_end(const char *tag, const char *backend_name,
                                  uint32_t stack_high_water_words)
{
    (void)tag;
    (void)backend_name;
    (void)stack_high_water_words;
}

void resource_profile_session_cancel(void) {}

#endif /* CONFIG_REPLAY_RESOURCE_PROFILE */
