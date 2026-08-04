/*
 * Optional UART smoke pass at replay firmware boot (before Wi-Fi).
 * Optional embedded-vector self-check when CONFIG_REPLAY_BOOT_SMOKE=y.
 */
#ifndef REPLAY_BOOT_SMOKE_H
#define REPLAY_BOOT_SMOKE_H

void replay_run_boot_smoke(void);

#endif
