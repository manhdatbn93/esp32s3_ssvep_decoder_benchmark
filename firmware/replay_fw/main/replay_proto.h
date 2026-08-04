/**
 * @file replay_proto.h
 * @brief Binary framing for Wi-Fi TCP dataset replay (PC client <-> ESP32-S3 server).
 *
 * Shared protocol between firmware and tools/replay/replay_proto.py.
 * Keep both in sync (magic, version, types, struct layouts).
 */
#ifndef REPLAY_PROTO_H
#define REPLAY_PROTO_H

#include <stdint.h>

#define REPLAY_MAGIC_U32 0x52425032u /* 'RBP2' little-endian */

#define REPLAY_VERSION_MAJOR 2u
#define REPLAY_VERSION_MINOR 0u

typedef enum {
  REPLAY_PKT_HELLO     = 1,
  REPLAY_PKT_HELLO_ACK = 2,
  REPLAY_PKT_START_RUN = 3,
  REPLAY_PKT_START_ACK = 4,
  REPLAY_PKT_TRIAL_DATA = 5,
  REPLAY_PKT_RESULT    = 6,
  REPLAY_PKT_END_RUN   = 7,
  REPLAY_PKT_END_ACK   = 8,
  REPLAY_PKT_ERROR     = 9,
} replay_pkt_type_t;

typedef enum {
  REPLAY_NORM_NONE            = 0,
  REPLAY_NORM_CHANNEL_ZSCORE  = 1,
  REPLAY_NORM_GLOBAL_MINMAX   = 2,
} replay_normalize_mode_t;

typedef enum {
  REPLAY_DECODER_CCA        = 0,
  REPLAY_DECODER_FBCCA      = 1,
  REPLAY_DECODER_MSI        = 2,
  REPLAY_DECODER_TRCA       = 3,
  REPLAY_DECODER_EEGNET_F32 = 4,
  REPLAY_DECODER_EEGNET_I8  = 5,
  REPLAY_DECODER_CCNN_F32   = 6,
  REPLAY_DECODER_CCNN_I8    = 7,
  REPLAY_DECODER_CONVCA_F32 = 8,
  REPLAY_DECODER_FBTCNN_F32 = 9,
  REPLAY_DECODER_FBTCNN_I8  = 10,
  REPLAY_DECODER_COUNT      = 11,
} replay_decoder_id_t;

/* Fixed wire header (little-endian). Total 16 bytes. */
typedef struct __attribute__((packed)) replay_wire_header {
  uint32_t magic;
  uint16_t version_major;
  uint16_t version_minor;
  uint16_t type;        /* replay_pkt_type_t */
  uint16_t flags;       /* reserved, 0 */
  uint32_t payload_len; /* bytes following header */
} replay_wire_header_t;

_Static_assert(sizeof(replay_wire_header_t) == 16, "replay header size");

/* START_RUN payload — 16 bytes on wire. */
typedef struct __attribute__((packed)) replay_start_payload {
  uint8_t  normalize_mode; /* replay_normalize_mode_t */
  uint8_t  onset_sample_0; /* default 38 */
  uint16_t window_length;  /* default 512 */
  uint16_t fs_hz;          /* 256 */
  uint8_t  n_channels;     /* 8 */
  uint8_t  decoder_id;     /* replay_decoder_id_t */
  uint8_t  reserved[8];
} replay_start_payload_t;

_Static_assert(sizeof(replay_start_payload_t) == 16, "replay_start_payload_t wire size");

/* Per-trial metadata sent before raw samples in TRIAL_DATA payload. */
typedef struct __attribute__((packed)) replay_trial_meta {
  char     subject[8];       /* e.g. "s1\0..." */
  uint8_t  class_id;         /* 0..11 */
  uint8_t  trial_idx;        /* 0..14 within class */
  uint16_t n_time;           /* e.g. 1114 */
  uint8_t  n_channels;       /* 8 */
  uint8_t  reserved0;
  uint16_t fs_hz;
  uint8_t  normalize_mode;
  uint8_t  reserved1[5];
} replay_trial_meta_t;

_Static_assert(sizeof(replay_trial_meta_t) == 22, "trial meta wire size");

/* RESULT payload: meta echo + 12 floats (logits or scores). */
typedef struct __attribute__((packed)) replay_result_payload {
  replay_trial_meta_t meta;
  uint8_t  decoder_id;
  uint8_t  status;         /* 0 = ok, nonzero = forward error */
  uint16_t reserved;
  uint32_t elapsed_us;     /* wall-clock microseconds for preprocess + forward */
  float    logits[12];
} replay_result_payload_t;

typedef struct __attribute__((packed)) replay_error_payload {
  uint32_t code;
  char     message[96];
} replay_error_payload_t;

/* Optional START_ACK payload: embedded TFLite flatbuffer identity. */
#define REPLAY_MODEL_IDENTITY_TAG_LEN     24
#define REPLAY_MODEL_IDENTITY_SYMBOL_LEN  48

typedef struct __attribute__((packed)) replay_model_identity_payload {
  uint8_t  requested_decoder_id;
  uint8_t  actual_decoder_id;
  uint8_t  input_dtype;   /* TfLiteType, e.g. 1=float32 */
  uint8_t  output_dtype;
  uint32_t model_size;
  uint32_t model_crc32;
  uint8_t  first16[16];
  uint8_t  last16[16];
  char     embed_tag[REPLAY_MODEL_IDENTITY_TAG_LEN];
  char     symbol[REPLAY_MODEL_IDENTITY_SYMBOL_LEN];
} replay_model_identity_payload_t;

_Static_assert(sizeof(replay_model_identity_payload_t) == 116,
               "replay_model_identity_payload_t wire size");

#define REPLAY_EPSILON_ZSCORE 1e-8f

#endif /* REPLAY_PROTO_H */
