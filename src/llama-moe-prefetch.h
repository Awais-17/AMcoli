/**
 * @file llama-moe-prefetch.h
 * @brief Speculative prefetch engine for MoE expert streaming.
 *
 * Internal header — not part of the public API.
 *
 * Phase 4 feature: Uses router logits from the current layer to
 * speculatively predict which experts will be needed in the next layer,
 * then issues prefetch hints (madvise/io_uring) to overlap disk I/O
 * with compute.
 *
 * Strategy:
 *   1. After computing router logits for layer L, top-k expert IDs are known
 *   2. Speculatively compute layer L+1 router logits using layer L's input
 *      (lightweight: just a matmul + softmax on the gate, always resident)
 *   3. Take top-k from speculative logits → prefetch those experts
 *   4. ~60-70% overlap accuracy between prediction and actual routing
 *
 * Fallback: wrong predictions cause no harm — the miss path is identical
 * to the standard synchronous disk load. Only a wasted read.
 */

#ifndef LLAMA_MOE_PREFETCH_H
#define LLAMA_MOE_PREFETCH_H

#include "amcoli.h"
#include "llama-disk-streamer.h"
#include "llama-moe-cache.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Prefetch Engine ─────────────────────────────────────────────────── */

/**
 * Prefetch engine state.
 * Tracks pending prefetches and their outcomes for statistics.
 */
struct moe_prefetch_engine {
    struct disk_streamer *streamer; /**< Borrowed pointer to disk streamer  */
    struct moe_cache     *cache;   /**< Borrowed pointer to expert cache   */

    int32_t  prefetch_depth;       /**< Layers ahead to prefetch (1 or 2)  */
    bool     enabled;              /**< Master enable flag                 */

    /* Pending prefetches: ring buffer of recently prefetched (layer, expert) pairs */
    struct {
        int32_t layer_id;
        int32_t expert_id;
        bool    was_used;          /**< Set to true if actually accessed   */
    } pending[256];                /**< Circular buffer                    */
    int32_t  pending_head;
    int32_t  pending_count;

    /* Statistics */
    int64_t  stat_prefetch_issued;
    int64_t  stat_prefetch_hit;    /**< Prefetched expert was actually used */
    int64_t  stat_prefetch_wasted; /**< Prefetched expert was evicted unused */
};

/**
 * Initialize the prefetch engine.
 *
 * @param engine    Output engine struct
 * @param streamer  Disk streamer to issue prefetch hints through
 * @param cache     Expert cache to check before prefetching
 * @param depth     Layers ahead to prefetch (0 = disabled)
 * @return          AMCOLI_OK or error code
 */
int32_t moe_prefetch_init(
    struct moe_prefetch_engine *engine,
    struct disk_streamer *streamer,
    struct moe_cache     *cache,
    int32_t               depth
);

/** Free prefetch engine resources. */
void moe_prefetch_free(struct moe_prefetch_engine *engine);

/**
 * Issue prefetch hints for predicted next-layer experts.
 *
 * @param engine      The prefetch engine
 * @param next_layer  The layer to prefetch for
 * @param expert_ids  Array of predicted expert IDs
 * @param n_experts   Length of expert_ids
 *
 * Only issues prefetch for experts NOT already in the cache.
 * Non-blocking.
 */
void moe_prefetch_issue(
    struct moe_prefetch_engine *engine,
    int32_t        next_layer,
    const int32_t *expert_ids,
    int32_t        n_experts
);

/**
 * Mark a prefetched expert as "used" (for accuracy statistics).
 * Call this when an expert is actually accessed during inference.
 */
void moe_prefetch_mark_used(
    struct moe_prefetch_engine *engine,
    int32_t layer_id,
    int32_t expert_id
);

/** Print prefetch statistics to stderr. */
void moe_prefetch_print_stats(const struct moe_prefetch_engine *engine);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_MOE_PREFETCH_H */
