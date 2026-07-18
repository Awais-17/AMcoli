/**
 * @file amcoli-sys-info.h
 * @brief Platform-specific system information helper.
 *
 * Detects total and available physical RAM and VRAM to auto-size
 * cache budgets. Supports Windows and Linux.
 */

#ifndef AMCOLI_SYS_INFO_H
#define AMCOLI_SYS_INFO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get total physical RAM in bytes.
 * Returns 0 if detection fails.
 */
int64_t amcoli_sys_get_total_ram(void);

/**
 * Get currently available physical RAM in bytes.
 * Returns 0 if detection fails.
 */
int64_t amcoli_sys_get_available_ram(void);

/**
 * Get total VRAM in bytes.
 * If CUDA/Vulkan are not linked, falls back to DXGI (Windows)
 * or returns 0.
 */
int64_t amcoli_sys_get_total_vram(void);

/**
 * Get currently available VRAM in bytes.
 */
int64_t amcoli_sys_get_available_vram(void);

/**
 * Fetch the hardware name of the primary GPU adapter.
 */
void amcoli_sys_get_gpu_name(char *name_out, size_t max_len);

/**
 * Print detected system specifications and recommend compatible MoE models
 * along with recommended quantization levels.
 */
void amcoli_print_recommendations(void);

#ifdef __cplusplus
}
#endif

#endif /* AMCOLI_SYS_INFO_H */
