/*
 * Optional boot self-check: run embedded vectors through every compiled backend.
 *
 * Controlled by CONFIG_REPLAY_BOOT_SMOKE (see Kconfig.projbuild).
 * Runs before Wi-Fi so TFLite + classical paths are verified even if Wi-Fi fails.
 *
 * Backend flags: same SMOKE_ENABLE_* CMake options as the rest of replay_fw.
 */
#include "sdkconfig.h"

#if CONFIG_REPLAY_BOOT_SMOKE

#include "esp_log.h"
#include "ssvep_dispatch.h"
#include <stddef.h>

static const char *TAG_BOOT = "replay_boot";

/* ── Classical backends ─────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_CCA
extern int  cca_smoke_init(void);
extern int  cca_smoke_run(const float *input, float *scores);
extern void cca_smoke_cleanup(void);
extern const int          cca_smoke_n_vectors;
extern const float *const cca_smoke_vectors[];
extern const int          cca_smoke_expected_argmax[];
extern const int          cca_smoke_has_expected_scores;
extern const float       *cca_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_FBCCA
extern int  fbcca_smoke_init(void);
extern int  fbcca_smoke_run(const float *input, float *scores);
extern void fbcca_smoke_cleanup(void);
extern const int          fbcca_smoke_n_vectors;
extern const float *const fbcca_smoke_vectors[];
extern const int          fbcca_smoke_expected_argmax[];
extern const int          fbcca_smoke_has_expected_scores;
extern const float       *fbcca_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_MSI
extern int  msi_smoke_init(void);
extern int  msi_smoke_run(const float *input, float *scores);
extern void msi_smoke_cleanup(void);
extern const int          msi_smoke_n_vectors;
extern const float *const msi_smoke_vectors[];
extern const int          msi_smoke_expected_argmax[];
extern const int          msi_smoke_has_expected_scores;
extern const float       *msi_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_TRCA
extern int  trca_smoke_init(void);
extern int  trca_smoke_run(const float *input, float *scores);
extern void trca_smoke_cleanup(void);
extern const int          trca_smoke_n_vectors;
extern const float *const trca_smoke_vectors[];
extern const int   *trca_smoke_expected_argmax;
extern const int    trca_smoke_has_expected_scores;
extern const float *trca_smoke_expected_scores_flat;
#endif

/* ── DL backends ───────────────────────────────────────────────── */

#ifdef SMOKE_ENABLE_EEGNET_F32
extern int  eegnet_f32_smoke_init(void);
extern int  eegnet_f32_smoke_run(const float *input, float *scores);
extern void eegnet_f32_smoke_cleanup(void);
extern const int          eegnet_f32_smoke_n_vectors;
extern const float *const eegnet_f32_smoke_vectors[];
extern const int          eegnet_f32_smoke_expected_argmax[];
extern const int          eegnet_f32_smoke_has_expected_scores;
extern const float       *eegnet_f32_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_EEGNET_INT8
extern int  eegnet_i8_smoke_init(void);
extern int  eegnet_i8_smoke_run(const float *input, float *scores);
extern void eegnet_i8_smoke_cleanup(void);
extern const int          eegnet_i8_smoke_n_vectors;
extern const float *const eegnet_i8_smoke_vectors[];
extern const int          eegnet_i8_smoke_expected_argmax[];
extern const int          eegnet_i8_smoke_has_expected_scores;
extern const float       *eegnet_i8_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_CCNN_F32
extern int  ccnn_f32_smoke_init(void);
extern int  ccnn_f32_smoke_run(const float *input, float *scores);
extern void ccnn_f32_smoke_cleanup(void);
extern const int          ccnn_f32_smoke_n_vectors;
extern const float *const ccnn_f32_smoke_vectors[];
extern const int          ccnn_f32_smoke_expected_argmax[];
extern const int          ccnn_f32_smoke_has_expected_scores;
extern const float       *ccnn_f32_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_CCNN_INT8
extern int  ccnn_i8_smoke_init(void);
extern int  ccnn_i8_smoke_run(const float *input, float *scores);
extern void ccnn_i8_smoke_cleanup(void);
extern const int          ccnn_i8_smoke_n_vectors;
extern const float *const ccnn_i8_smoke_vectors[];
extern const int          ccnn_i8_smoke_expected_argmax[];
extern const int          ccnn_i8_smoke_has_expected_scores;
extern const float       *ccnn_i8_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_CONVCA_F32
extern int  convca_f32_smoke_init(void);
extern int  convca_f32_smoke_run(const float *input, float *scores);
extern void convca_f32_smoke_cleanup(void);
extern const int          convca_f32_smoke_n_vectors;
extern const float *const convca_f32_smoke_vectors[];
extern const int          convca_f32_smoke_expected_argmax[];
extern const int          convca_f32_smoke_has_expected_scores;
extern const float       *convca_f32_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_CONVCA_INT8
extern int  convca_i8_smoke_init(void);
extern int  convca_i8_smoke_run(const float *input, float *scores);
extern void convca_i8_smoke_cleanup(void);
extern const int          convca_i8_smoke_n_vectors;
extern const float *const convca_i8_smoke_vectors[];
extern const int          convca_i8_smoke_expected_argmax[];
extern const int          convca_i8_smoke_has_expected_scores;
extern const float       *convca_i8_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_FBTCNN_INT8
extern int  fbtcnn_i8_smoke_init(void);
extern int  fbtcnn_i8_smoke_run(const float *input, float *scores);
extern void fbtcnn_i8_smoke_cleanup(void);
extern const int          fbtcnn_i8_smoke_n_vectors;
extern const float *const fbtcnn_i8_smoke_vectors[];
extern const int          fbtcnn_i8_smoke_expected_argmax[];
extern const int          fbtcnn_i8_smoke_has_expected_scores;
extern const float       *fbtcnn_i8_smoke_expected_scores_flat;
#endif

#ifdef SMOKE_ENABLE_FBTCNN_F32
extern int  fbtcnn_f32_smoke_init(void);
extern int  fbtcnn_f32_smoke_run(const float *input, float *scores);
extern void fbtcnn_f32_smoke_cleanup(void);
extern const int          fbtcnn_f32_smoke_n_vectors;
extern const float *const fbtcnn_f32_smoke_vectors[];
extern const int          fbtcnn_f32_smoke_expected_argmax[];
extern const int          fbtcnn_f32_smoke_has_expected_scores;
extern const float       *fbtcnn_f32_smoke_expected_scores_flat;
#endif

#endif /* CONFIG_REPLAY_BOOT_SMOKE */

#include "replay_boot_smoke.h"

void replay_run_boot_smoke(void)
{
#if !CONFIG_REPLAY_BOOT_SMOKE
    return;
#else
    smoke_backend_t backends[SMOKE_MAX_BACKENDS];
    smoke_result_t  results[SMOKE_MAX_BACKENDS];
    int n = 0;

    ESP_LOGI(TAG_BOOT, "========================================");
    ESP_LOGI(TAG_BOOT, "  BOOT SELF-CHECK (embedded vectors)");
    ESP_LOGI(TAG_BOOT, "========================================");

#ifdef SMOKE_ENABLE_CCA
    backends[n++] = (smoke_backend_t){
        .name = "CCA", .init = cca_smoke_init, .run = cca_smoke_run,
        .cleanup = cca_smoke_cleanup, .n_vectors = cca_smoke_n_vectors,
        .vectors = cca_smoke_vectors, .expected_argmax = cca_smoke_expected_argmax,
        .has_expected_scores = cca_smoke_has_expected_scores,
        .expected_scores_flat = cca_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_FBCCA
    backends[n++] = (smoke_backend_t){
        .name = "FBCCA", .init = fbcca_smoke_init, .run = fbcca_smoke_run,
        .cleanup = fbcca_smoke_cleanup, .n_vectors = fbcca_smoke_n_vectors,
        .vectors = fbcca_smoke_vectors, .expected_argmax = fbcca_smoke_expected_argmax,
        .has_expected_scores = fbcca_smoke_has_expected_scores,
        .expected_scores_flat = fbcca_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_MSI
    backends[n++] = (smoke_backend_t){
        .name = "MSI", .init = msi_smoke_init, .run = msi_smoke_run,
        .cleanup = msi_smoke_cleanup, .n_vectors = msi_smoke_n_vectors,
        .vectors = msi_smoke_vectors, .expected_argmax = msi_smoke_expected_argmax,
        .has_expected_scores = msi_smoke_has_expected_scores,
        .expected_scores_flat = msi_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_TRCA
    backends[n++] = (smoke_backend_t){
        .name = "TRCA", .init = trca_smoke_init, .run = trca_smoke_run,
        .cleanup = trca_smoke_cleanup, .n_vectors = trca_smoke_n_vectors,
        .vectors = trca_smoke_vectors, .expected_argmax = trca_smoke_expected_argmax,
        .has_expected_scores = trca_smoke_has_expected_scores,
        .expected_scores_flat = trca_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_EEGNET_F32
    backends[n++] = (smoke_backend_t){
        .name = "EEGNet-f32", .init = eegnet_f32_smoke_init, .run = eegnet_f32_smoke_run,
        .cleanup = eegnet_f32_smoke_cleanup, .n_vectors = eegnet_f32_smoke_n_vectors,
        .vectors = eegnet_f32_smoke_vectors, .expected_argmax = eegnet_f32_smoke_expected_argmax,
        .has_expected_scores = eegnet_f32_smoke_has_expected_scores,
        .expected_scores_flat = eegnet_f32_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_EEGNET_INT8
    backends[n++] = (smoke_backend_t){
        .name = "EEGNet-int8", .init = eegnet_i8_smoke_init, .run = eegnet_i8_smoke_run,
        .cleanup = eegnet_i8_smoke_cleanup, .n_vectors = eegnet_i8_smoke_n_vectors,
        .vectors = eegnet_i8_smoke_vectors, .expected_argmax = eegnet_i8_smoke_expected_argmax,
        .has_expected_scores = eegnet_i8_smoke_has_expected_scores,
        .expected_scores_flat = eegnet_i8_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_CCNN_F32
    backends[n++] = (smoke_backend_t){
        .name = "CCNN-f32", .init = ccnn_f32_smoke_init, .run = ccnn_f32_smoke_run,
        .cleanup = ccnn_f32_smoke_cleanup, .n_vectors = ccnn_f32_smoke_n_vectors,
        .vectors = ccnn_f32_smoke_vectors, .expected_argmax = ccnn_f32_smoke_expected_argmax,
        .has_expected_scores = ccnn_f32_smoke_has_expected_scores,
        .expected_scores_flat = ccnn_f32_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_CCNN_INT8
    backends[n++] = (smoke_backend_t){
        .name = "CCNN-int8", .init = ccnn_i8_smoke_init, .run = ccnn_i8_smoke_run,
        .cleanup = ccnn_i8_smoke_cleanup, .n_vectors = ccnn_i8_smoke_n_vectors,
        .vectors = ccnn_i8_smoke_vectors, .expected_argmax = ccnn_i8_smoke_expected_argmax,
        .has_expected_scores = ccnn_i8_smoke_has_expected_scores,
        .expected_scores_flat = ccnn_i8_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_CONVCA_F32
    backends[n++] = (smoke_backend_t){
        .name = "ConvCA-f32", .init = convca_f32_smoke_init, .run = convca_f32_smoke_run,
        .cleanup = convca_f32_smoke_cleanup, .n_vectors = convca_f32_smoke_n_vectors,
        .vectors = convca_f32_smoke_vectors, .expected_argmax = convca_f32_smoke_expected_argmax,
        .has_expected_scores = convca_f32_smoke_has_expected_scores,
        .expected_scores_flat = convca_f32_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_CONVCA_INT8
    backends[n++] = (smoke_backend_t){
        .name = "ConvCA-int8 (diag)", .init = convca_i8_smoke_init, .run = convca_i8_smoke_run,
        .cleanup = convca_i8_smoke_cleanup, .n_vectors = convca_i8_smoke_n_vectors,
        .vectors = convca_i8_smoke_vectors, .expected_argmax = convca_i8_smoke_expected_argmax,
        .has_expected_scores = convca_i8_smoke_has_expected_scores,
        .expected_scores_flat = convca_i8_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_FBTCNN_INT8
    backends[n++] = (smoke_backend_t){
        .name = "FBtCNN-int8", .init = fbtcnn_i8_smoke_init, .run = fbtcnn_i8_smoke_run,
        .cleanup = fbtcnn_i8_smoke_cleanup, .n_vectors = fbtcnn_i8_smoke_n_vectors,
        .vectors = fbtcnn_i8_smoke_vectors, .expected_argmax = fbtcnn_i8_smoke_expected_argmax,
        .has_expected_scores = fbtcnn_i8_smoke_has_expected_scores,
        .expected_scores_flat = fbtcnn_i8_smoke_expected_scores_flat,
    };
#endif
#ifdef SMOKE_ENABLE_FBTCNN_F32
    backends[n++] = (smoke_backend_t){
        .name = "FBtCNN-f32", .init = fbtcnn_f32_smoke_init, .run = fbtcnn_f32_smoke_run,
        .cleanup = fbtcnn_f32_smoke_cleanup, .n_vectors = fbtcnn_f32_smoke_n_vectors,
        .vectors = fbtcnn_f32_smoke_vectors, .expected_argmax = fbtcnn_f32_smoke_expected_argmax,
        .has_expected_scores = fbtcnn_f32_smoke_has_expected_scores,
        .expected_scores_flat = fbtcnn_f32_smoke_expected_scores_flat,
    };
#endif

    if (n == 0) {
        ESP_LOGW(TAG_BOOT, "No backends enabled — skip boot smoke (set SMOKE_ENABLE_*)");
        return;
    }

    ESP_LOGI(TAG_BOOT, "Registered %d backend(s) for boot smoke", n);
    (void)smoke_run_all(backends, n, results);

    ESP_LOGI(TAG_BOOT, "========================================");
    ESP_LOGI(TAG_BOOT, "  BOOT SMOKE DONE — starting Wi-Fi");
    ESP_LOGI(TAG_BOOT, "========================================");
#endif
}
