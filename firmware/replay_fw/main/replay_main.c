/*
 * Dataset replay firmware — ESP32-S3.
 *
 * Connects to Wi-Fi, starts a TCP server, and waits for a PC client
 * to stream raw 12JFPM trials. Each trial is preprocessed on-device
 * and run through the selected decoder backend. Results (12 float
 * scores/logits) are returned over TCP for host-side comparison.
 *
 * One image contains one staged backend and fold. Stage the frozen
 * reference artifacts, copy sdkconfig.measurement to sdkconfig, then
 * build with idf.py. Full commands are in docs/firmware-and-replay.md.
 * Do not run idf.py set-target after that copy.
 */
#include "esp_log.h"
#include "replay_embed_tag.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "replay_boot_smoke.h"
#include "replay_tcp.h"
#include "wifi_connect.h"

static const char *TAG = "replay_app";

/*
 * lwIP BSD sockets use pthread TLS via sys_thread_sem_get(). With
 * CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY, xTaskCreate() may place
 * large stacks in PSRAM, which breaks recv/send. Keep the TCP server
 * stack in internal DRAM (static .bss).
 */
#define REPLAY_TCP_STACK_BYTES (16 * 1024)
static StackType_t replay_tcp_stack[REPLAY_TCP_STACK_BYTES / sizeof(StackType_t)];
static StaticTask_t replay_tcp_tcb;

void app_main(void) {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "  SSVEP Dataset Replay Firmware v2.0");
  ESP_LOGI(TAG, "  ESP32-S3 / 12JFPM Benchmark");
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Embedded export tag: %s", REPLAY_EMBED_TARGET_TAG);
  ESP_LOGI(TAG, "Staged fold tag: %s", REPLAY_EMBED_TARGET_TAG);

  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  replay_run_boot_smoke();

  ESP_ERROR_CHECK(replay_wifi_init_and_connect());

  xTaskCreateStaticPinnedToCore(
      replay_tcp_server_task,
      "replay_tcp",
      REPLAY_TCP_STACK_BYTES,
      NULL,
      5,
      replay_tcp_stack,
      &replay_tcp_tcb,
      0);
  ESP_LOGI(TAG, "replay_tcp_server_task started (internal stack %d B, core 0)",
           REPLAY_TCP_STACK_BYTES);
}
