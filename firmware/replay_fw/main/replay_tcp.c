/*
 * TCP server for dataset replay: framed binary protocol, one client at a time.
 *
 * Receives raw trials from PC, preprocesses (crop + z-score), runs the
 * selected backend, and returns 12 float scores/logits.
 *
 * Backend dispatch reuses the same init/run/cleanup functions already
 * proven in the smoke firmware. Classical backends use *_forward_*;
 * DL backends use *_smoke_* wrappers (which handle TFLite + preprocessing).
 */
#include "replay_tcp.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "replay_proto.h"
#include "replay_embed_tag.h"
#include "preprocess_12jfpm.h"
#include "backend_identity.h"
#include "model_identity.h"
#include "resource_profile.h"
#include "sdkconfig.h"
#if defined(SMOKE_ENABLE_EEGNET_F32) || defined(SMOKE_ENABLE_EEGNET_INT8)
#include "eegnet_model_identity.h"
#endif

/* Classical backend forward declarations */
#ifdef SMOKE_ENABLE_CCA
#include "cca_forward.h"
#endif
#ifdef SMOKE_ENABLE_FBCCA
#include "fbcca_forward.h"
#endif
#ifdef SMOKE_ENABLE_MSI
#include "msi_forward.h"
#endif
#ifdef SMOKE_ENABLE_TRCA
#include "trca_forward.h"
#endif

/* DL backend forward declarations */
#ifdef SMOKE_ENABLE_EEGNET_F32
extern int  eegnet_f32_smoke_init(void);
extern int  eegnet_f32_smoke_run(const float *input, float *scores);
extern void eegnet_f32_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_EEGNET_INT8
extern int  eegnet_i8_smoke_init(void);
extern int  eegnet_i8_smoke_run(const float *input, float *scores);
extern void eegnet_i8_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_CCNN_F32
extern int  ccnn_f32_smoke_init(void);
extern int  ccnn_f32_smoke_run(const float *input, float *scores);
extern void ccnn_f32_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_CCNN_INT8
extern int  ccnn_i8_smoke_init(void);
extern int  ccnn_i8_smoke_run(const float *input, float *scores);
extern void ccnn_i8_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_CONVCA_F32
extern int  convca_f32_smoke_init(void);
extern int  convca_f32_smoke_run(const float *input, float *scores);
extern void convca_f32_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_FBTCNN_F32
extern int  fbtcnn_f32_smoke_init(void);
extern int  fbtcnn_f32_smoke_run(const float *input, float *scores);
extern void fbtcnn_f32_smoke_cleanup(void);
#endif
#ifdef SMOKE_ENABLE_FBTCNN_INT8
extern int  fbtcnn_i8_smoke_init(void);
extern int  fbtcnn_i8_smoke_run(const float *input, float *scores);
extern void fbtcnn_i8_smoke_cleanup(void);
#endif

static const char *TAG = "replay";

/* ── Decoder name LUT (always available, no ifdef) ─────────────── */

static const char *decoder_name(int id) {
  static const char *NAMES[REPLAY_DECODER_COUNT] = {
    [REPLAY_DECODER_CCA]        = "CCA",
    [REPLAY_DECODER_FBCCA]      = "FBCCA",
    [REPLAY_DECODER_MSI]        = "MSI",
    [REPLAY_DECODER_TRCA]       = "TRCA",
    [REPLAY_DECODER_EEGNET_F32] = "EEGNet-f32",
    [REPLAY_DECODER_EEGNET_I8]  = "EEGNet-int8",
    [REPLAY_DECODER_CCNN_F32]   = "CCNN-f32",
    [REPLAY_DECODER_CCNN_I8]    = "CCNN-int8",
    [REPLAY_DECODER_CONVCA_F32] = "ConvCA-f32",
    [REPLAY_DECODER_FBTCNN_F32] = "FBtCNN-f32",
    [REPLAY_DECODER_FBTCNN_I8]  = "FBtCNN-int8",
  };
  if (id >= 0 && id < REPLAY_DECODER_COUNT && NAMES[id])
    return NAMES[id];
  return "???";
}

/* ── Backend table ──────────────────────────────────────────────── */

typedef struct {
  const char *name;
  int  (*init)(void);
  int  (*run)(const float *input_tc, float *scores_out);
  void (*cleanup)(void);
} replay_backend_t;

static const replay_backend_t BACKEND_TABLE[REPLAY_DECODER_COUNT] = {
#ifdef SMOKE_ENABLE_CCA
  [REPLAY_DECODER_CCA] = { "CCA", cca_forward_init, cca_forward_run, cca_forward_cleanup },
#endif
#ifdef SMOKE_ENABLE_FBCCA
  [REPLAY_DECODER_FBCCA] = { "FBCCA", fbcca_forward_init, fbcca_forward_run, fbcca_forward_cleanup },
#endif
#ifdef SMOKE_ENABLE_MSI
  [REPLAY_DECODER_MSI] = { "MSI", msi_forward_init, msi_forward_run, msi_forward_cleanup },
#endif
#ifdef SMOKE_ENABLE_TRCA
  [REPLAY_DECODER_TRCA] = { "TRCA", trca_forward_init, trca_forward_run, trca_forward_cleanup },
#endif
#ifdef SMOKE_ENABLE_EEGNET_F32
  [REPLAY_DECODER_EEGNET_F32] = { "EEGNet-f32", eegnet_f32_smoke_init, eegnet_f32_smoke_run, eegnet_f32_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_EEGNET_INT8
  [REPLAY_DECODER_EEGNET_I8] = { "EEGNet-int8", eegnet_i8_smoke_init, eegnet_i8_smoke_run, eegnet_i8_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_CCNN_F32
  [REPLAY_DECODER_CCNN_F32] = { "CCNN-f32", ccnn_f32_smoke_init, ccnn_f32_smoke_run, ccnn_f32_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_CCNN_INT8
  [REPLAY_DECODER_CCNN_I8] = { "CCNN-int8", ccnn_i8_smoke_init, ccnn_i8_smoke_run, ccnn_i8_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_CONVCA_F32
  [REPLAY_DECODER_CONVCA_F32] = { "ConvCA-f32", convca_f32_smoke_init, convca_f32_smoke_run, convca_f32_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_FBTCNN_F32
  [REPLAY_DECODER_FBTCNN_F32] = { "FBtCNN-f32", fbtcnn_f32_smoke_init, fbtcnn_f32_smoke_run, fbtcnn_f32_smoke_cleanup },
#endif
#ifdef SMOKE_ENABLE_FBTCNN_INT8
  [REPLAY_DECODER_FBTCNN_I8] = { "FBtCNN-int8", fbtcnn_i8_smoke_init, fbtcnn_i8_smoke_run, fbtcnn_i8_smoke_cleanup },
#endif
};

/* ── Session state ──────────────────────────────────────────────── */

static int      s_active_decoder = -1;
static float    s_x_tc[512 * 8];
static float    s_logits[12];

/* Crop buffer for raw trial → [window, channels]. ~16 KB static. */
static float    s_crop[512 * 8];

/* Per-session counters for summary logging */
static uint32_t s_trial_count;
static uint32_t s_trial_ok;
static uint32_t s_trial_err;
static uint64_t s_latency_sum_us;
static uint32_t s_latency_min_us;
static uint32_t s_latency_max_us;

/* RES_PROFILE: log pre/post heap only for the first successful forward per session. */
static int s_logged_pre_first_infer;
static int s_logged_post_first_infer;

/* ── TCP helpers ────────────────────────────────────────────────── */

static int recv_all(int sock, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t got = 0;
  while (got < len) {
    int r = recv(sock, p + got, len - got, 0);
    if (r < 0) { ESP_LOGE(TAG, "recv errno=%d", errno); return -1; }
    if (r == 0) { ESP_LOGW(TAG, "peer closed"); return -1; }
    got += (size_t)r;
  }
  return 0;
}

static int send_all(int sock, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t sent = 0;
  while (sent < len) {
    int w = send(sock, p + sent, len - sent, 0);
    if (w < 0) { ESP_LOGE(TAG, "send errno=%d", errno); return -1; }
    sent += (size_t)w;
  }
  return 0;
}

static void send_error_pkt(int sock, uint32_t code, const char *msg) {
  replay_error_payload_t ep;
  memset(&ep, 0, sizeof(ep));
  ep.code = code;
  if (msg) strncpy(ep.message, msg, sizeof(ep.message) - 1);
  replay_wire_header_t oh = {
    .magic = REPLAY_MAGIC_U32,
    .version_major = REPLAY_VERSION_MAJOR,
    .version_minor = REPLAY_VERSION_MINOR,
    .type = REPLAY_PKT_ERROR,
    .flags = 0,
    .payload_len = sizeof(ep)
  };
  send_all(sock, &oh, sizeof(oh));
  send_all(sock, &ep, sizeof(ep));
}

static void send_header(int sock, uint16_t type, uint32_t payload_len) {
  replay_wire_header_t h = {
    .magic = REPLAY_MAGIC_U32,
    .version_major = REPLAY_VERSION_MAJOR,
    .version_minor = REPLAY_VERSION_MINOR,
    .type = type,
    .flags = 0,
    .payload_len = payload_len
  };
  send_all(sock, &h, sizeof(h));
}

static int fill_backend_identity(int decoder_id,
                                 replay_model_identity_payload_t *out) {
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  out->requested_decoder_id = (uint8_t)decoder_id;
  out->actual_decoder_id = (uint8_t)decoder_id;

#if defined(SMOKE_ENABLE_EEGNET_INT8) || defined(SMOKE_ENABLE_EEGNET_F32)
  {
    eegnet_model_identity_t id;
    int rc = -1;
#ifdef SMOKE_ENABLE_EEGNET_INT8
    if (decoder_id == REPLAY_DECODER_EEGNET_I8)
      rc = eegnet_i8_model_identity(&id);
#endif
#ifdef SMOKE_ENABLE_EEGNET_F32
    if (decoder_id == REPLAY_DECODER_EEGNET_F32)
      rc = eegnet_f32_model_identity(&id);
#endif
    if (rc == 0) {
      out->model_size = (uint32_t)id.len;
      out->model_crc32 = id.crc32;
      memcpy(out->first16, id.first16, sizeof(out->first16));
      memcpy(out->last16, id.last16, sizeof(out->last16));
      out->input_dtype = (uint8_t)id.input_dtype;
      out->output_dtype = (uint8_t)id.output_dtype;
      strncpy(out->symbol, id.symbol ? id.symbol : "", sizeof(out->symbol) - 1);
      strncpy(out->embed_tag, REPLAY_EMBED_TARGET_TAG, sizeof(out->embed_tag) - 1);
      return 0;
    }
  }
#endif

  /* Staged analytical / neural identity for every selected backend. */
  out->model_size = ssvep_backend_identity_size();
  out->model_crc32 = ssvep_backend_identity_crc32();
  memcpy(out->first16, ssvep_backend_identity_first16(), sizeof(out->first16));
  memcpy(out->last16, ssvep_backend_identity_last16(), sizeof(out->last16));
  out->input_dtype = 1;
  out->output_dtype = 1;
  strncpy(out->symbol, ssvep_backend_identity_name(), sizeof(out->symbol) - 1);
  strncpy(out->embed_tag, REPLAY_EMBED_TARGET_TAG, sizeof(out->embed_tag) - 1);
  return 0;
}

static void send_start_ack(int sock, int decoder_id) {
  replay_model_identity_payload_t ident;
  if (fill_backend_identity(decoder_id, &ident) == 0) {
    char fhex[33], lhex[33];
    model_identity_hex16(ident.first16, 16, 0, fhex, sizeof(fhex));
    model_identity_hex16(ident.last16, 16, 0, lhex, sizeof(lhex));
    const char *req_name = decoder_name(decoder_id);
    ESP_LOGI(TAG, "[fw:model] requested_backend=%s", req_name);
    ESP_LOGI(TAG, "[fw:model] actual_backend=%s", decoder_name((int)ident.actual_decoder_id));
    ESP_LOGI(TAG, "[fw:model] staged_backend=%s", ssvep_backend_identity_name());
    ESP_LOGI(TAG, "[fw:model] tag=%s", ident.embed_tag);
    ESP_LOGI(TAG, "[fw:model] symbol=%s", ident.symbol);
    ESP_LOGI(TAG, "[fw:model] size=%" PRIu32, ident.model_size);
    ESP_LOGI(TAG, "[fw:model] crc32=0x%08" PRIx32, ident.model_crc32);
    ESP_LOGI(TAG, "[fw:model] first16=%s", fhex);
    ESP_LOGI(TAG, "[fw:model] last16=%s", lhex);
    ESP_LOGI(TAG, "[fw:model] input_dtype=%s",
             ident.input_dtype == 1 ? "float32" : "?");
    ESP_LOGI(TAG, "[fw:model] output_dtype=%s",
             ident.output_dtype == 1 ? "float32" : "?");
    send_header(sock, REPLAY_PKT_START_ACK, (uint32_t)sizeof(ident));
    send_all(sock, &ident, sizeof(ident));
  } else {
    send_header(sock, REPLAY_PKT_START_ACK, 0);
  }
}

/* ── Preprocessing ──────────────────────────────────────────────── */

static int preprocess_trial(const float *raw, int n_time, int n_ch,
                             int onset, int window,
                             replay_normalize_mode_t mode, float *out_tc) {
  if (n_ch <= 0 || window <= 0) return -1;
  if (onset < 0 || onset + window > n_time) {
    ESP_LOGE(TAG, "bad crop: onset=%d window=%d n_time=%d", onset, window, n_time);
    return -2;
  }
  if (window * n_ch > (int)(sizeof(s_crop) / sizeof(s_crop[0]))) {
    ESP_LOGE(TAG, "window*ch too large");
    return -3;
  }

  for (int t = 0; t < window; t++)
    for (int c = 0; c < n_ch; c++)
      s_crop[t * n_ch + c] = raw[(onset + t) * n_ch + c];

  if (mode == REPLAY_NORM_NONE) {
    memcpy(out_tc, s_crop, (size_t)(window * n_ch) * sizeof(float));
    return 0;
  }
  if (mode == REPLAY_NORM_CHANNEL_ZSCORE) {
    for (int c = 0; c < n_ch; c++) {
      double sum = 0.0;
      for (int t = 0; t < window; t++)
        sum += (double)s_crop[t * n_ch + c];
      double mean = sum / (double)window;
      double var = 0.0;
      for (int t = 0; t < window; t++) {
        double d = (double)s_crop[t * n_ch + c] - mean;
        var += d * d;
      }
      double std = sqrt(var / (double)window);
      if (std < (double)REPLAY_EPSILON_ZSCORE) std = (double)REPLAY_EPSILON_ZSCORE;
      for (int t = 0; t < window; t++)
        out_tc[t * n_ch + c] = (float)(((double)s_crop[t * n_ch + c] - mean) / std);
    }
    return 0;
  }
  ESP_LOGE(TAG, "unknown norm mode %d", (int)mode);
  return -4;
}

/* ── Backend lifecycle ──────────────────────────────────────────── */

static void cleanup_active_decoder(void) {
  if (s_active_decoder >= 0 && s_active_decoder < REPLAY_DECODER_COUNT) {
    const replay_backend_t *b = &BACKEND_TABLE[s_active_decoder];
    if (b->cleanup) {
      ESP_LOGI(TAG, "cleanup decoder %d (%s)", s_active_decoder, b->name ? b->name : "?");
      b->cleanup();
    }
  }
  s_active_decoder = -1;
}

static int init_decoder(int decoder_id) {
  if (decoder_id < 0 || decoder_id >= REPLAY_DECODER_COUNT)
    return -1;
  const replay_backend_t *b = &BACKEND_TABLE[decoder_id];
  if (!b->name || !b->init || !b->run)
    return -1;
  cleanup_active_decoder();
  ESP_LOGI(TAG, "init decoder %d (%s)", decoder_id, b->name);
#if CONFIG_REPLAY_RESOURCE_PROFILE
  resource_profile_session_pre_init(TAG, b->name);
#endif
  int rc = b->init();
  if (rc != 0) {
#if CONFIG_REPLAY_RESOURCE_PROFILE
    resource_profile_session_cancel();
#endif
    ESP_LOGE(TAG, "init failed rc=%d", rc);
    return rc;
  }
#if CONFIG_REPLAY_RESOURCE_PROFILE
  resource_profile_session_post_init(TAG, b->name);
#endif
  s_active_decoder = decoder_id;
  return 0;
}

static int _argmax12(const float *v) {
  int best = 0;
  for (int i = 1; i < 12; i++)
    if (v[i] > v[best]) best = i;
  return best;
}

/* ── Client handler ─────────────────────────────────────────────── */

static int handle_client(int sock) {
  replay_start_payload_t sess;
  memset(&sess, 0, sizeof(sess));
  sess.normalize_mode = REPLAY_NORM_CHANNEL_ZSCORE;
  sess.onset_sample_0 = 38;
  sess.window_length = 512;
  sess.fs_hz = 256;
  sess.n_channels = 8;
  int session_ok = 0;

  for (;;) {
    replay_wire_header_t hdr;
    if (recv_all(sock, &hdr, sizeof(hdr)) != 0) break;
    if (hdr.magic != REPLAY_MAGIC_U32) {
      ESP_LOGE(TAG, "bad magic 0x%08" PRIx32, hdr.magic);
      break;
    }
    uint32_t plen = hdr.payload_len;
    if (plen > 4u * 1024u * 1024u) {
      ESP_LOGE(TAG, "payload too large");
      break;
    }
    uint8_t *payload = NULL;
    if (plen > 0) {
      payload = heap_caps_malloc(plen, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      if (!payload) payload = malloc(plen);
      if (!payload) {
        ESP_LOGE(TAG, "oom payload %u", (unsigned)plen);
        break;
      }
      if (recv_all(sock, payload, plen) != 0) { free(payload); break; }
    }

    switch (hdr.type) {
      case REPLAY_PKT_HELLO: {
        send_header(sock, REPLAY_PKT_HELLO_ACK, 0);
        break;
      }

      case REPLAY_PKT_START_RUN: {
        if (plen < sizeof(replay_start_payload_t)) {
          send_error_pkt(sock, 3, "START_RUN payload too short");
          session_ok = 0;
        } else {
          memcpy(&sess, payload, sizeof(sess));
          int dec = (int)sess.decoder_id;
          if (dec < 0 || dec >= REPLAY_DECODER_COUNT || !BACKEND_TABLE[dec].name) {
            session_ok = 0;
            ESP_LOGW(TAG, "unknown/disabled decoder_id %d", dec);
            send_error_pkt(sock, 2, "unknown or disabled decoder_id");
          } else {
            int rc = init_decoder(dec);
            if (rc != 0) {
              session_ok = 0;
              send_error_pkt(sock, 5, "decoder init failed");
            } else {
              session_ok = 1;
              s_trial_count = s_trial_ok = s_trial_err = 0;
              s_latency_sum_us = 0;
              s_latency_min_us = UINT32_MAX;
              s_latency_max_us = 0;
              s_logged_pre_first_infer = 0;
              s_logged_post_first_infer = 0;
              ESP_LOGI(TAG, "──── SESSION START ────────────────────");
              ESP_LOGI(TAG, "  backend : %s (id=%d)", decoder_name(dec), dec);
              ESP_LOGI(TAG, "  norm=%u  onset=%u  window=%u  fs=%u  ch=%u",
                       (unsigned)sess.normalize_mode,
                       (unsigned)sess.onset_sample_0,
                       (unsigned)sess.window_length,
                       (unsigned)sess.fs_hz,
                       (unsigned)sess.n_channels);
              send_start_ack(sock, dec);
            }
          }
        }
        free(payload);
        continue;
      }

      case REPLAY_PKT_TRIAL_DATA: {
        if (!session_ok) {
          send_error_pkt(sock, 4, "session not started");
          free(payload);
          continue;
        }
        if (plen < sizeof(replay_trial_meta_t)) {
          ESP_LOGE(TAG, "trial too small");
          free(payload);
          break;
        }
        replay_trial_meta_t meta;
        memcpy(&meta, payload, sizeof(meta));
        size_t raw_bytes = plen - sizeof(meta);
        if (raw_bytes % sizeof(float) != 0) {
          ESP_LOGE(TAG, "raw size not float-aligned");
          free(payload);
          break;
        }
        int n_floats = (int)(raw_bytes / sizeof(float));
        int n_ch = meta.n_channels > 0 ? (int)meta.n_channels : (int)sess.n_channels;
        if (n_ch <= 0) n_ch = 8;
        int n_time = meta.n_time > 0 ? (int)meta.n_time : (n_floats / n_ch);
        const float *raw = (const float *)(payload + sizeof(meta));
        if (n_time * n_ch != n_floats) {
          n_time = n_floats / n_ch;
        }

        int64_t t0 = esp_timer_get_time();

        memset(s_logits, 0, sizeof(s_logits));
        uint8_t status = 0;

        int pr = preprocess_trial(raw, n_time, n_ch,
                                  (int)sess.onset_sample_0,
                                  (int)sess.window_length,
                                  (replay_normalize_mode_t)sess.normalize_mode,
                                  s_x_tc);
        if (pr != 0) {
          ESP_LOGE(TAG, "preprocess failed %d", pr);
          status = 1;
        } else {
#if CONFIG_REPLAY_RESOURCE_PROFILE
          if (!s_logged_pre_first_infer) {
            resource_profile_heap_t hp;
            resource_profile_heap_read(&hp);
            resource_profile_log_line(TAG, "pre_first_infer",
                                      decoder_name(s_active_decoder), &hp);
            s_logged_pre_first_infer = 1;
          }
#endif
          const replay_backend_t *b = &BACKEND_TABLE[s_active_decoder];
          int fwd = b->run(s_x_tc, s_logits);
          if (fwd != 0) {
            ESP_LOGE(TAG, "forward failed dec=%d rc=%d", s_active_decoder, fwd);
            status = 2;
          }
#if CONFIG_REPLAY_RESOURCE_PROFILE
          if (fwd == 0) {
            if (!s_logged_post_first_infer) {
              resource_profile_heap_t hi;
              resource_profile_heap_read(&hi);
              resource_profile_log_line(TAG, "post_first_infer",
                                        decoder_name(s_active_decoder), &hi);
              s_logged_post_first_infer = 1;
            }
            resource_profile_session_sample_after_work();
          }
#endif
        }

        int64_t t1 = esp_timer_get_time();
        uint32_t elapsed_us = (uint32_t)(t1 - t0);

        replay_result_payload_t res;
        memset(&res, 0, sizeof(res));
        memcpy(&res.meta, &meta, sizeof(meta));
        res.decoder_id = (uint8_t)s_active_decoder;
        res.status = status;
        res.elapsed_us = elapsed_us;
        memcpy(res.logits, s_logits, sizeof(res.logits));

        send_header(sock, REPLAY_PKT_RESULT, sizeof(res));
        send_all(sock, &res, sizeof(res));

        s_trial_count++;
        if (status == 0) {
          s_trial_ok++;
          s_latency_sum_us += elapsed_us;
          if (elapsed_us < s_latency_min_us) s_latency_min_us = elapsed_us;
          if (elapsed_us > s_latency_max_us) s_latency_max_us = elapsed_us;
        } else {
          s_trial_err++;
        }
        int argmax = _argmax12(s_logits);
        const char *match = (argmax == (int)meta.class_id) ? "OK" : "MISS";
        ESP_LOGI(TAG, "  [%3u] %s c=%2u t=%2u argmax=%2d %s  %.1f ms",
                 (unsigned)s_trial_count,
                 decoder_name(s_active_decoder),
                 (unsigned)meta.class_id,
                 (unsigned)meta.trial_idx,
                 argmax, match,
                 (float)elapsed_us / 1000.0f);

        free(payload);
        continue;
      }

      case REPLAY_PKT_END_RUN: {
        ESP_LOGI(TAG, "──── SESSION END ──────────────────────");
#if CONFIG_REPLAY_RESOURCE_PROFILE
        if (session_ok && s_active_decoder >= 0) {
          UBaseType_t hwm = uxTaskGetStackHighWaterMark(NULL);
          resource_profile_session_end(TAG, decoder_name(s_active_decoder),
                                       (uint32_t)hwm);
        }
#endif
        if (s_trial_count > 0) {
          float avg_ms = (s_trial_ok > 0)
              ? (float)s_latency_sum_us / (float)s_trial_ok / 1000.0f
              : 0.0f;
          ESP_LOGI(TAG, "  backend : %s", decoder_name(s_active_decoder));
          ESP_LOGI(TAG, "  trials  : %u total, %u ok, %u err",
                   (unsigned)s_trial_count, (unsigned)s_trial_ok, (unsigned)s_trial_err);
          if (s_trial_ok > 0)
            ESP_LOGI(TAG, "  latency : avg %.1f ms, min %.1f ms, max %.1f ms",
                     avg_ms,
                     (float)s_latency_min_us / 1000.0f,
                     (float)s_latency_max_us / 1000.0f);
        } else {
          ESP_LOGI(TAG, "  (no trials in this session)");
        }
        ESP_LOGI(TAG, "──────────────────────────────────────");
        cleanup_active_decoder();
        session_ok = 0;
        send_header(sock, REPLAY_PKT_END_ACK, 0);
        free(payload);
        return 0;
      }

      default:
        ESP_LOGW(TAG, "unknown type %u", (unsigned)hdr.type);
        break;
    }
    free(payload);
  }

  cleanup_active_decoder();
  return -1;
}

/* ── TCP server task ────────────────────────────────────────────── */

void replay_tcp_server_task(void *pv) {
  int port = CONFIG_REPLAY_TCP_PORT;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  int ls = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
  if (ls < 0) {
    ESP_LOGE(TAG, "socket listen failed");
    vTaskDelete(NULL);
    return;
  }
  int opt = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  if (bind(ls, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    ESP_LOGE(TAG, "bind port %d errno=%d", port, errno);
    close(ls);
    vTaskDelete(NULL);
    return;
  }
  listen(ls, 1);
  ESP_LOGI(TAG, "listening on port %d", port);

  ESP_LOGI(TAG, "Available backends:");
  int n_avail = 0;
  for (int i = 0; i < REPLAY_DECODER_COUNT; i++) {
    if (BACKEND_TABLE[i].name && BACKEND_TABLE[i].run) {
      ESP_LOGI(TAG, "  [%2d] %s", i, BACKEND_TABLE[i].name);
      n_avail++;
    }
  }
  if (n_avail == 0) {
    ESP_LOGW(TAG, "  (no backends compiled — rebuild with SMOKE_ENABLE_* flags)");
  } else {
    ESP_LOGI(TAG, "  %d backend(s) ready", n_avail);
  }
  ESP_LOGI(TAG, "Waiting for client connection...");

  for (;;) {
    struct sockaddr_in peer;
    socklen_t plen = sizeof(peer);
    int cs = accept(ls, (struct sockaddr *)&peer, &plen);
    if (cs < 0) {
      vTaskDelay(pdMS_TO_TICKS(200));
      continue;
    }
    {
      char ip_str[INET_ADDRSTRLEN] = {0};
      inet_ntoa_r(peer.sin_addr, ip_str, sizeof(ip_str));
      ESP_LOGI(TAG, "──────────────────────────────────────");
      ESP_LOGI(TAG, "client connected from %s:%u", ip_str, ntohs(peer.sin_port));
    }
    handle_client(cs);
    shutdown(cs, SHUT_RDWR);
    close(cs);
    ESP_LOGI(TAG, "client disconnected, waiting for next");
  }
}
