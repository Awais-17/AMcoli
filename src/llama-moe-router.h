/**
 * @file llama-moe-router.h
 * @brief Generic MoE router configuration extraction from GGUF metadata.
 *
 * Internal header — not part of the public API.
 *
 * Extracts architecture-generic MoE configuration from any GGUF file by:
 *   1. Reading standard metadata keys (expert_count, expert_used_count)
 *   2. Scanning tensor names for expert patterns (routed + shared)
 *   3. Detecting which layers are MoE layers vs. dense layers
 *
 * This approach is model-agnostic: it handles Mixtral (no shared experts),
 * Qwen-MoE (some shared), DeepSeek-V3/GLM-5.2 (always shared) without
 * any architecture-specific code paths.
 */

#ifndef LLAMA_MOE_ROUTER_H
#define LLAMA_MOE_ROUTER_H

#include "amcoli.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Router Config Builder ───────────────────────────────────────────── */

/**
 * Extract MoE configuration from a GGUF file.
 *
 * @param file_data      Pointer to mmap'd GGUF file data
 * @param file_size      Size of the file in bytes
 * @param config_out     Output: populated MoE configuration
 * @return               AMCOLI_OK, AMCOLI_ERR_NOT_MOE, or error code
 *
 * This function:
 *   1. Reads GGUF header to get KV count and tensor count
 *   2. Scans KV pairs for:
 *      - *.expert_count → n_expert
 *      - *.expert_used_count → n_expert_used (top-k)
 *      - *.block_count → n_total_layers
 *   3. Scans tensor names for:
 *      - ffn_gate_exp / ffn_up_exp / ffn_down_exp → routed experts
 *      - ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp → shared experts
 *   4. Determines which layers are MoE layers by checking which layer
 *      indices have expert tensors
 *   5. Computes per-expert byte sizes and total model breakdown
 *
 * If expert_count is 0 or not found, returns AMCOLI_ERR_NOT_MOE.
 */
int32_t moe_router_extract_config(
    const void *file_data,
    size_t      file_size,
    struct amcoli_moe_config *config_out
);

/**
 * Free any dynamically allocated fields in a moe_config.
 * (Currently just moe_layer_ids.)
 */
void moe_config_free(struct amcoli_moe_config *config);

/**
 * Validate that a moe_config is internally consistent.
 *
 * @param config  The config to validate
 * @return        AMCOLI_OK if valid, AMCOLI_ERR_INVALID_PARAMS if not
 *
 * Checks:
 *   - n_expert > 0
 *   - n_expert_used > 0 and <= n_expert
 *   - n_moe_layers > 0 and <= n_total_layers
 *   - expert_total_bytes > 0
 *   - moe_layer_ids are within [0, n_total_layers)
 */
int32_t moe_config_validate(const struct amcoli_moe_config *config);

/**
 * Compute memory budget requirements for a given configuration.
 *
 * @param config        The MoE config
 * @param vram_slots    Number of VRAM cache slots
 * @param ram_slots     Number of RAM cache slots
 * @param vram_needed   Output: VRAM bytes needed for cache
 * @param ram_needed    Output: RAM bytes needed (resident + cache)
 * @param disk_needed   Output: Total disk bytes needed (full model)
 *
 * Useful for validating user-declared budgets before allocating.
 */
void moe_config_compute_budget(
    const struct amcoli_moe_config *config,
    int32_t  vram_slots,
    int32_t  ram_slots,
    int64_t *vram_needed,
    int64_t *ram_needed,
    int64_t *disk_needed
);

/**
 * Print a human-readable summary of the MoE configuration.
 * Used by the `--model-info` CLI flag.
 */
void moe_config_print(const struct amcoli_moe_config *config);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_MOE_ROUTER_H */
