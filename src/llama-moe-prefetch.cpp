/**
 * @file llama-moe-prefetch.cpp
 * @brief Speculative prefetch engine implementation.
 *
 * Issues prefetch hints for experts predicted to be needed in upcoming layers.
 * Uses a circular buffer to track pending prefetches and measure accuracy.
 */

#include "llama-moe-prefetch.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int32_t moe_prefetch_init(
    struct moe_prefetch_engine *engine,
    struct disk_streamer *streamer,
    struct moe_cache     *cache,
    int32_t               depth
) {
    if (!engine) return AMCOLI_ERR_INVALID_PARAMS;

    memset(engine, 0, sizeof(*engine));
    engine->streamer       = streamer;
    engine->cache          = cache;
    engine->prefetch_depth = depth;
    engine->enabled        = (depth > 0 && streamer != NULL && cache != NULL);
    engine->pending_head   = 0;
    engine->pending_count  = 0;

    return AMCOLI_OK;
}

void moe_prefetch_free(struct moe_prefetch_engine *engine) {
    if (!engine) return;
    /* No dynamic allocations — just zero out */
    memset(engine, 0, sizeof(*engine));
}

void moe_prefetch_issue(
    struct moe_prefetch_engine *engine,
    int32_t        next_layer,
    const int32_t *expert_ids,
    int32_t        n_experts
) {
    if (!engine || !engine->enabled || !expert_ids || n_experts <= 0) return;

    for (int32_t i = 0; i < n_experts; i++) {
        int32_t eid = expert_ids[i];

        /* Skip if already cached */
        int32_t tier;
        void *data;
        size_t size;
        if (moe_cache_lookup(engine->cache, next_layer, eid, &tier, &data, &size)) {
            continue; /* Already in cache, no need to prefetch */
        }

        /* Issue the prefetch hint */
        disk_streamer_prefetch_expert(engine->streamer, next_layer, eid);
        engine->stat_prefetch_issued++;

        /* Track in pending buffer (circular) */
        int32_t idx = (engine->pending_head + engine->pending_count) % 256;
        if (engine->pending_count >= 256) {
            /* Buffer full — oldest entry is lost (considered wasted) */
            if (!engine->pending[engine->pending_head].was_used) {
                engine->stat_prefetch_wasted++;
            }
            engine->pending_head = (engine->pending_head + 1) % 256;
            engine->pending_count--;
        }

        engine->pending[idx].layer_id  = next_layer;
        engine->pending[idx].expert_id = eid;
        engine->pending[idx].was_used  = false;
        engine->pending_count++;
    }
}

void moe_prefetch_mark_used(
    struct moe_prefetch_engine *engine,
    int32_t layer_id,
    int32_t expert_id
) {
    if (!engine || !engine->enabled) return;

    /* Search pending buffer for this (layer, expert) */
    for (int32_t i = 0; i < engine->pending_count; i++) {
        int32_t idx = (engine->pending_head + i) % 256;
        if (engine->pending[idx].layer_id == layer_id &&
            engine->pending[idx].expert_id == expert_id &&
            !engine->pending[idx].was_used)
        {
            engine->pending[idx].was_used = true;
            engine->stat_prefetch_hit++;
            return;
        }
    }
}

void moe_prefetch_print_stats(const struct moe_prefetch_engine *engine) {
    if (!engine) return;

    fprintf(stderr, "\n=== AMcoli Prefetch Statistics ===\n");
    fprintf(stderr, "  Enabled:    %s\n", engine->enabled ? "yes" : "no");
    fprintf(stderr, "  Depth:      %d layers ahead\n", engine->prefetch_depth);
    fprintf(stderr, "  Issued:     %lld\n", (long long)engine->stat_prefetch_issued);
    fprintf(stderr, "  Hits:       %lld\n", (long long)engine->stat_prefetch_hit);
    fprintf(stderr, "  Wasted:     %lld\n", (long long)engine->stat_prefetch_wasted);

    if (engine->stat_prefetch_issued > 0) {
        double accuracy = 100.0 * (double)engine->stat_prefetch_hit /
                          (double)engine->stat_prefetch_issued;
        fprintf(stderr, "  Accuracy:   %.1f%%\n", accuracy);
    }
    fprintf(stderr, "=================================\n\n");
}
