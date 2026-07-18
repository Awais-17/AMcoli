/**
 * @file llama-moe-cache.h
 * @brief Two-tier expert cache (VRAM + RAM) with pluggable eviction.
 *
 * Internal header — not part of the public API.
 *
 * Architecture:
 *   ┌──────────────────────────────────┐
 *   │       VRAM Slot Pool (Tier 1)    │  ← GPU-resident hot experts
 *   │  [slot0: L3/E5] [slot1: L3/E2]  │
 *   ├──────────────────────────────────┤
 *   │       RAM Slot Pool (Tier 2)     │  ← CPU-pinned warm experts
 *   │  [slot0: L7/E1] [slot1: L2/E4]  │
 *   └──────────────────────────────────┘
 *
 * Each slot holds one complete expert's weight tensors (gate + up + down).
 * Eviction is per-slot, not per-layer.
 *
 * Thread safety:
 *   - Read operations (lookup) are lock-free via atomic generation counters
 *   - Write operations (insert/evict) hold a per-tier spinlock
 */

#ifndef LLAMA_MOE_CACHE_H
#define LLAMA_MOE_CACHE_H

#include "amcoli.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ───────────────────────────────────────────────────────── */

#define MOE_CACHE_INVALID_SLOT   (-1)
#define MOE_CACHE_MAX_LAYERS     256
#define MOE_CACHE_MAX_EXPERTS    32768  /* GLM-5.2 has 21504 */

/* ── Cache Slot ──────────────────────────────────────────────────────── */

/** State of a cache slot. */
enum moe_slot_state {
    MOE_SLOT_EMPTY    = 0, /**< Slot is free                              */
    MOE_SLOT_LOADING  = 1, /**< Slot is being filled from disk            */
    MOE_SLOT_READY    = 2, /**< Slot contains valid expert data           */
    MOE_SLOT_EVICTING = 3, /**< Slot is being evicted                     */
};

/**
 * A single cache slot in either the VRAM or RAM pool.
 *
 * Memory layout: the `data` pointer points to a contiguous buffer of
 * `data_size` bytes containing [gate_weights | up_weights | down_weights].
 */
struct moe_cache_slot {
    int32_t  layer_id;           /**< Layer this expert belongs to (-1 = empty)  */
    int32_t  expert_id;          /**< Expert index within the layer (-1 = empty) */
    int32_t  state;              /**< moe_slot_state enum                        */

    void    *data;               /**< Pointer to expert weight data              */
    size_t   data_size;          /**< Size of expert data in bytes               */

    /* LRU tracking */
    int64_t  last_access_tick;   /**< Monotonic tick at last access              */

    /* LFU tracking */
    int64_t  access_count;       /**< Total accesses since insertion             */

    /* Doubly-linked list pointers for LRU ordering */
    int32_t  lru_prev;           /**< Previous slot index in LRU list (-1 = head)*/
    int32_t  lru_next;           /**< Next slot index in LRU list (-1 = tail)    */
};

/* ── Slot Pool (one per tier) ────────────────────────────────────────── */

/** Which memory tier this pool manages. */
enum moe_cache_tier {
    MOE_TIER_VRAM = 0,
    MOE_TIER_RAM  = 1,
};

/**
 * A pool of cache slots for one tier.
 *
 * The slot array is pre-allocated at init time. `data_arena` is a single
 * contiguous allocation divided equally among slots.
 */
struct moe_slot_pool {
    enum moe_cache_tier tier;

    int32_t  n_slots;            /**< Total slots in this pool                   */
    int32_t  n_used;             /**< Currently occupied slots                   */
    struct moe_cache_slot *slots;/**< Array of n_slots slots                     */

    /* Memory arena: one contiguous block, each slot gets (arena_size / n_slots) */
    void    *data_arena;         /**< Backing memory for expert data             */
    size_t   arena_size;         /**< Total arena size in bytes                  */
    size_t   slot_data_size;     /**< Bytes per slot = expert_total_bytes        */

    /* LRU list head/tail (indices into slots array) */
    int32_t  lru_head;           /**< Most recently used slot                    */
    int32_t  lru_tail;           /**< Least recently used slot (eviction victim) */

    /* Monotonic tick counter for access ordering */
    int64_t  tick;

    /* Fast lookup: hash map from (layer_id, expert_id) → slot index */
    int32_t *lookup_table;       /**< Hash table of slot indices                 */
    int32_t  lookup_capacity;    /**< Capacity of lookup_table                   */

    /* Statistics */
    int64_t  stat_hits;
    int64_t  stat_misses;
    int64_t  stat_evictions;
};

/* ── Cache (combines both tiers) ─────────────────────────────────────── */

/**
 * The complete two-tier expert cache.
 */
struct moe_cache {
    struct moe_slot_pool  vram_pool;
    struct moe_slot_pool  ram_pool;

    int32_t  eviction_policy;    /**< amcoli_eviction_policy enum               */

    /* Per-layer statistics for diagnostics */
    struct {
        int64_t hits;
        int64_t misses;
    } layer_stats[MOE_CACHE_MAX_LAYERS];

    int32_t  n_layers;           /**< Number of MoE layers being tracked        */
};

/* ── API ─────────────────────────────────────────────────────────────── */

/**
 * Initialize the two-tier cache.
 *
 * @param cache           Output cache struct
 * @param cache_params    Cache configuration (slots, budgets, policy)
 * @param expert_bytes    Size of one expert's data in bytes
 * @param n_moe_layers    Number of MoE layers in the model
 * @return                AMCOLI_OK or error code
 *
 * Allocates the VRAM and RAM arenas and initializes all slot metadata.
 * If vram_slots or ram_slots are 0, computes them from the budgets.
 */
int32_t moe_cache_init(
    struct moe_cache *cache,
    const struct amcoli_cache_params *cache_params,
    size_t  expert_bytes,
    int32_t n_moe_layers
);

/** Free all cache resources. */
void moe_cache_free(struct moe_cache *cache);

/**
 * Look up an expert in the cache.
 *
 * @param cache      The cache
 * @param layer_id   Layer index
 * @param expert_id  Expert index
 * @param tier_out   If found, receives the tier (MOE_TIER_VRAM or MOE_TIER_RAM)
 * @param data_out   If found, receives pointer to expert data
 * @param size_out   If found, receives size of expert data
 * @return           true if found (cache hit), false if not (cache miss)
 *
 * On hit, updates the LRU/LFU tracking for the slot.
 */
bool moe_cache_lookup(
    struct moe_cache *cache,
    int32_t  layer_id,
    int32_t  expert_id,
    int32_t *tier_out,
    void   **data_out,
    size_t  *size_out
);

/**
 * Insert an expert into the cache.
 *
 * @param cache      The cache
 * @param tier       Which tier to insert into
 * @param layer_id   Layer index
 * @param expert_id  Expert index
 * @param src_data   Expert weight data to copy into the slot
 * @param src_size   Size of src_data in bytes
 * @param data_out   Receives pointer to the cached copy
 * @return           AMCOLI_OK, or error code if eviction/insert fails
 *
 * If the target tier is full, evicts the coldest slot first.
 * For VRAM inserts, the data is copied via a staging buffer.
 */
int32_t moe_cache_insert(
    struct moe_cache *cache,
    int32_t  tier,
    int32_t  layer_id,
    int32_t  expert_id,
    const void *src_data,
    size_t   src_size,
    void   **data_out
);

/**
 * Evict the coldest expert from the specified tier.
 *
 * @param cache   The cache
 * @param tier    Tier to evict from
 * @return        The slot index that was freed, or MOE_CACHE_INVALID_SLOT
 */
int32_t moe_cache_evict(struct moe_cache *cache, int32_t tier);

/**
 * Get the total hit rate across both tiers.
 */
double moe_cache_hit_rate(const struct moe_cache *cache);

/**
 * Dump cache statistics to stderr.
 */
void moe_cache_print_stats(const struct moe_cache *cache);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_MOE_CACHE_H */
