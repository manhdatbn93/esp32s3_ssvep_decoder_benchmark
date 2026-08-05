/*
 * Minimal Wi-Fi STA helper (ESP-IDF 5.x pattern).
 * Credentials from Kconfig REPLAY_WIFI_SSID / REPLAY_WIFI_PASSWORD.
 */
#include "wifi_connect.h"

#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_types.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "sdkconfig.h"

static const char *TAG = "replay_wifi";

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

static void event_handler(void *arg, esp_event_base_t base, int32_t id,
                           void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *e = (wifi_event_sta_disconnected_t *)data;
    ESP_LOGW(TAG, "disconnected reason=%u ssid=%.*s, retry",
             (unsigned)e->reason, e->ssid_len, (const char *)e->ssid);
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
    ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&e->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

esp_err_t replay_wifi_init_and_connect(void) {
  s_wifi_event_group = xEventGroupCreate();

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  esp_event_handler_instance_t inst_any;
  esp_event_handler_instance_t inst_got_ip;
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, &inst_any));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, &inst_got_ip));

  wifi_config_t wifi_config = {0};
  strncpy((char *)wifi_config.sta.ssid, CONFIG_REPLAY_WIFI_SSID,
          sizeof(wifi_config.sta.ssid) - 1);
  strncpy((char *)wifi_config.sta.password, CONFIG_REPLAY_WIFI_PASSWORD,
          sizeof(wifi_config.sta.password) - 1);
  /* Accept WPA/WPA2/WPA3 mixed APs. A hard WPA2-only threshold can fail on
   * some home routers even when the SSID is visible and auth briefly succeeds. */
  wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA_PSK;
  wifi_config.sta.pmf_cfg.capable = true;
  wifi_config.sta.pmf_cfg.required = false;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "connecting to %s ...", CONFIG_REPLAY_WIFI_SSID);
  xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE,
                      pdTRUE, portMAX_DELAY);
  return ESP_OK;
}
