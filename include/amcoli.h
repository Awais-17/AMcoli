/**
 * @file amcoli.h
 * @brief AMcoli — Universal MoE Disk-Streaming Inference Engine
 *
 * Public C API for AMcoli's disk-streaming MoE prototype.
 * Provides: two-tier expert cache, disk streaming, generic MoE config
 * extraction, and live inference statistics.
 *
 * Usage:
 *   1. Create amcoli_params with desired configuration
 *   2. Call amcoli_context_create() with a GGUF model path
 *   3. Future llama.cpp integration can call amcoli_ensure_expert() from
 *      the model forward path when routed expert weights are needed.
 *   4. Query stats with amcoli_get_stats()
 *   5. Free with amcoli_context_free()
 */

#ifndef AMCOLI_H
#define AMCOLI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ─────────────────────────────────────────────────────────── */

#define AMCOLI_VERSION_MAJOR 0
#define AMCOLI_VERSION_MINOR 1
#define AMCOLI_VERSION_PATCH 0
#define AMCOLI_VERSION_STRING "0.1.0"

/* ── Enums ───────────────────────────────────────────────────────────── */

/** Cache eviction policy. */
enum amcoli_eviction_policy {
    AMCOLI_EVICT_LRU  = 0, /**< Least Recently Used (default)            */
    AMCOLI_EVICT_LFU  = 1, /**< Least Frequently Used                    */
    AMCOLI_EVICT_SLRU = 2, /**< Segmented LRU (probationary + protected) */
};

/** Disk I/O backend. */
enum amcoli_io_backend {
    AMCOLI_IO_MMAP     = 0, /**< mmap + madvise (default, portable)      */
    AMCOLI_IO_IOURING  = 1, /**< io_uring (Linux 5.6+, experimental)     */
};

/** Error codes. */
enum amcoli_error {
    AMCOLI_OK                   =  0,
    AMCOLI_ERR_INVALID_PARAMS   = -1,
    AMCOLI_ERR_NOT_MOE          = -2, /**< Model has no MoE layers        */
    AMCOLI_ERR_BUDGET_EXCEEDED  = -3, /**< Model can't fit in declared budget */
    AMCOLI_ERR_MMAP_FAILED      = -4,
    AMCOLI_ERR_IO_FAILED        = -5,
    AMCOLI_ERR_OUT_OF_MEMORY    = -6,
    AMCOLI_ERR_UNSUPPORTED      = -7, /**< Feature not compiled in        */
};

/* ── MoE Configuration (read-only, extracted from GGUF) ──────────────── */

/**
 * Architecture-generic MoE configuration extracted from GGUF metadata.
 * Populated by amcoli_context_create() after parsing the model.
 */
struct amcoli_moe_config {
    int32_t  n_expert;           /**< Total routed experts per MoE layer    */
    int32_t  n_expert_used;      /**< Top-k experts activated per token     */
    bool     has_shared_experts; /**< Model uses shared (always-on) experts */
    int32_t  n_shared_expert;    /**< Number of shared expert groups        */
    int32_t  n_moe_layers;       /**< Count of layers that are MoE          */
    int32_t  n_total_layers;     /**< Total transformer layers              */
    int32_t *moe_layer_ids;      /**< Array of MoE layer indices [n_moe_layers] */

    /* Per-expert weight sizes (bytes, post-quantization) */
    int64_t  expert_bytes_gate;  /**< Size of one expert's gate projection  */
    int64_t  expert_bytes_up;    /**< Size of one expert's up projection    */
    int64_t  expert_bytes_down;  /**< Size of one expert's down projection  */
    int64_t  expert_total_bytes; /**< gate + up + down for one expert       */

    /* Total model breakdown */
    int64_t  resident_bytes;     /**< Dense params always in RAM (attn, embed, shared exp) */
    int64_t  routed_bytes;       /**< Total routed expert bytes on disk     */
};

/* ── Cache Configuration ─────────────────────────────────────────────── */

/**
 * Two-tier expert cache parameters.
 * Set budget to -1 for auto-detection, 0 to disable a tier.
 */
struct amcoli_cache_params {
    int64_t  vram_budget_bytes;  /**< Max VRAM for expert cache (-1 = auto) */
    int64_t  ram_budget_bytes;   /**< Max RAM for expert cache (-1 = auto)  */
    int32_t  eviction_policy;    /**< amcoli_eviction_policy enum value     */
    int32_t  vram_slots;         /**< Max experts in VRAM (0 = auto-compute)*/
    int32_t  ram_slots;          /**< Max experts in RAM (0 = auto-compute) */
};

/** Returns sensible defaults: auto-detect budgets, LRU eviction. */
struct amcoli_cache_params amcoli_cache_params_default(void);

/* ── Disk Streamer Configuration ─────────────────────────────────────── */

/**
 * Disk I/O configuration for expert streaming.
 */
struct amcoli_disk_params {
    const char *model_path;      /**< Path to GGUF file on disk (NVMe recommended) */
    int32_t     io_backend;      /**< amcoli_io_backend enum value          */
    int32_t     prefetch_depth;  /**< Layers ahead to prefetch (0 = off)    */
    bool        allow_mlock;     /**< Allow mlocking resident params        */
};

/** Returns sensible defaults: mmap backend, no prefetch. */
struct amcoli_disk_params amcoli_disk_params_default(void);

/* ── Combined Parameters ─────────────────────────────────────────────── */

/**
 * Top-level AMcoli parameter block.
 */
struct amcoli_params {
    struct amcoli_cache_params  cache;
    struct amcoli_disk_params   disk;
    bool   print_stats_on_exit; /**< Dump stats summary to stderr on free   */
    int32_t verbosity;          /**< 0 = silent, 1 = info, 2 = debug        */
};

/** Returns sensible defaults for all params. */
struct amcoli_params amcoli_params_default(void);

/* ── Runtime Statistics ──────────────────────────────────────────────── */

/**
 * Live inference statistics (requirement F7).
 * All rates/averages are computed since the last reset.
 */
struct amcoli_stats {
    /* Throughput */
    double   tok_per_sec_cold;   /**< tok/s during cold-cache phase         */
    double   tok_per_sec_warm;   /**< tok/s after cache warmup              */
    double   tok_per_sec_current;/**< Instantaneous tok/s (last 10 tokens)  */

    /* Cache performance */
    int64_t  cache_hits_vram;    /**< Expert fetches served from VRAM       */
    int64_t  cache_hits_ram;     /**< Expert fetches served from RAM        */
    int64_t  cache_misses;       /**< Expert fetches requiring disk I/O     */
    double   cache_hit_rate;     /**< (vram_hits + ram_hits) / total        */
    int64_t  evictions_vram;     /**< Number of VRAM evictions              */
    int64_t  evictions_ram;      /**< Number of RAM evictions               */

    /* I/O performance */
    double   disk_wait_ms_avg;   /**< Average ms waiting for disk per miss  */
    double   disk_wait_ms_p99;   /**< 99th percentile disk wait             */
    int64_t  disk_bytes_read;    /**< Total bytes read from disk            */
    int64_t  prefetch_hits;      /**< Prefetches that were actually used    */
    int64_t  prefetch_wasted;    /**< Prefetches that were evicted unused   */

    /* Memory */
    int64_t  peak_ram_bytes;     /**< Peak RAM usage                        */
    int64_t  peak_vram_bytes;    /**< Peak VRAM usage                       */
    int64_t  current_ram_bytes;  /**< Current RAM usage                     */
    int64_t  current_vram_bytes; /**< Current VRAM usage                    */

    /* Tokens */
    int64_t  n_tokens_total;     /**< Total tokens processed                */
    double   elapsed_sec;        /**< Wall-clock time since context creation */
};

/* ── Opaque Context ──────────────────────────────────────────────────── */

struct amcoli_context;

/* ── Lifecycle ───────────────────────────────────────────────────────── */

/**
 * Create an AMcoli context for disk-streaming inference.
 *
 * @param params   Configuration (cache sizes, eviction policy, disk I/O, etc.)
 * @param err_out  On failure, receives an amcoli_error code. May be NULL.
 * @return         Opaque context, or NULL on failure.
 *
 * This function:
 *   1. Opens and mmaps the GGUF file
 *   2. Extracts MoE configuration from GGUF metadata
 *   3. Builds the expert offset index
 *   4. Allocates the two-tier cache slot pools
 *   5. Validates the memory budget
 *
 * If the model is not an MoE model, returns NULL with AMCOLI_ERR_NOT_MOE.
 * If the model can't fit within the declared budget, returns NULL with
 * AMCOLI_ERR_BUDGET_EXCEEDED and logs a diagnostic.
 */
struct amcoli_context *amcoli_context_create(
    const struct amcoli_params *params,
    int32_t *err_out
);

/** Free all resources associated with the context. */
void amcoli_context_free(struct amcoli_context *ctx);

/* ── Queries ─────────────────────────────────────────────────────────── */

/** Get the extracted MoE configuration (read-only). */
const struct amcoli_moe_config *amcoli_get_moe_config(
    const struct amcoli_context *ctx
);

/** Snapshot current statistics. Thread-safe. */
struct amcoli_stats amcoli_get_stats(const struct amcoli_context *ctx);

/** Reset all counters to zero (e.g. after warmup phase). */
void amcoli_reset_stats(struct amcoli_context *ctx);

/** Get cache occupancy statistics. */
void amcoli_get_cache_info(
    const struct amcoli_context *ctx,
    int32_t *vram_used,
    int32_t *vram_total,
    int32_t *ram_used,
    int32_t *ram_total
);

/* ── Expert Cache Operations ─────────────────────────────────────────── */

/**
 * Ensure the specified expert is available for compute.
 *
 * @param ctx       AMcoli context
 * @param layer_id  Transformer layer index
 * @param expert_id Expert index within the layer
 * @param data_out  Receives pointer to the expert weights (valid until next
 *                  call that may evict — caller must use before releasing)
 * @param size_out  Receives the byte size of the expert data
 * @return          AMCOLI_OK on success, error code on failure
 *
 * Cache lookup order: VRAM → RAM → Disk.
 * On a miss, the expert is loaded from disk into the appropriate cache tier,
 * potentially evicting a cold expert.
 */
int32_t amcoli_ensure_expert(
    struct amcoli_context *ctx,
    int32_t  layer_id,
    int32_t  expert_id,
    void   **data_out,
    size_t  *size_out
);

/**
 * Hint that the given experts will likely be needed soon (prefetch).
 * Non-blocking. Issues madvise/io_uring reads in the background.
 *
 * @param ctx         AMcoli context
 * @param layer_id    Target layer
 * @param expert_ids  Array of expert indices to prefetch
 * @param n_experts   Length of expert_ids array
 */
void amcoli_prefetch_experts(
    struct amcoli_context *ctx,
    int32_t        layer_id,
    const int32_t *expert_ids,
    int32_t        n_experts
);

/* ── Utility ─────────────────────────────────────────────────────────── */

/** Print MoE config summary to stderr. */
void amcoli_print_moe_config(const struct amcoli_moe_config *config);

/** Print stats summary to stderr. */
void amcoli_print_stats(const struct amcoli_stats *stats);

/** Return human-readable error string. */
const char *amcoli_error_string(int32_t err);

#ifdef __cplusplus
}
#endif

#endif /* AMCOLI_H */
