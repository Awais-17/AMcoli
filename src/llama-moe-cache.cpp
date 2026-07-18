/**
 * @file llama-moe-cache.cpp
 * @brief Two-tier expert cache implementation (VRAM + RAM slots, LRU/LFU).
 *
 * This is the core memory management layer of AMcoli. It maintains two pools
 * of fixed-size slots (one per memory tier) and provides O(1) lookup via a
 * hash table plus O(1) eviction via a doubly-linked LRU list.
 *
 * Design notes:
 * - Slots are pre-allocated at init. No runtime malloc in the hot path.
 * - The data arena is a single contiguous allocation. Each slot owns a
 *   fixed-size region within it, so "inserting" an expert is a memcpy
 *   into the slot's region — no pointer chasing.
 * - The hash table uses open addressing with linear probing. The key is
 *   a combined (layer_id, expert_id) packed into a 64-bit integer.
 * - LRU is maintained as a doubly-linked list threaded through the slots
 *   array. On access, the slot is moved to the head. Eviction pops the tail.
 */

#include "llama-moe-cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ── Hash table helpers ──────────────────────────────────────────────── */

/**
 * Pack (layer_id, expert_id) into a single 64-bit key for hash lookup.
 */
static inline uint64_t make_key(int32_t layer_id, int32_t expert_id) {
    return ((uint64_t)(uint32_t)layer_id << 32) | (uint64_t)(uint32_t)expert_id;
}

/**
 * FNV-1a inspired hash for 64-bit keys.
 * Chosen for speed + decent distribution on sequential expert IDs.
 */
static inline uint32_t hash_key(uint64_t key, int32_t capacity) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)(key % (uint64_t)capacity);
}

/* ── Slot pool internal helpers ──────────────────────────────────────── */

/**
 * Initialize a single slot pool (VRAM or RAM tier).
 */
static int32_t pool_init(
    struct moe_slot_pool *pool,
    enum moe_cache_tier   tier,
    int32_t               n_slots,
    size_t                expert_bytes
) {
    if (n_slots <= 0) {
        /* Tier disabled */
        memset(pool, 0, sizeof(*pool));
        pool->tier = tier;
        pool->lru_head = MOE_CACHE_INVALID_SLOT;
        pool->lru_tail = MOE_CACHE_INVALID_SLOT;
        return AMCOLI_OK;
    }

    pool->tier          = tier;
    pool->n_slots       = n_slots;
    pool->n_used        = 0;
    pool->slot_data_size = expert_bytes;
    pool->arena_size    = (size_t)n_slots * expert_bytes;
    pool->tick          = 0;
    pool->lru_head      = MOE_CACHE_INVALID_SLOT;
    pool->lru_tail      = MOE_CACHE_INVALID_SLOT;
    pool->stat_hits     = 0;
    pool->stat_misses   = 0;
    pool->stat_evictions = 0;

    /* Allocate slot metadata array */
    pool->slots = (struct moe_cache_slot *)calloc(
        (size_t)n_slots, sizeof(struct moe_cache_slot)
    );
    if (!pool->slots) {
        return AMCOLI_ERR_OUT_OF_MEMORY;
    }

    /* Allocate data arena
     * For VRAM tier, this would be a GPU allocation in production.
     * For now, we use CPU memory for both tiers (GPU integration in Phase 2).
     */
    pool->data_arena = calloc(1, pool->arena_size);
    if (!pool->data_arena) {
        free(pool->slots);
        pool->slots = NULL;
        return AMCOLI_ERR_OUT_OF_MEMORY;
    }

    /* Initialize each slot */
    for (int32_t i = 0; i < n_slots; i++) {
        struct moe_cache_slot *s = &pool->slots[i];
        s->layer_id         = -1;
        s->expert_id        = -1;
        s->state            = MOE_SLOT_EMPTY;
        s->data             = (char *)pool->data_arena + (size_t)i * expert_bytes;
        s->data_size        = expert_bytes;
        s->last_access_tick = 0;
        s->access_count     = 0;
        s->lru_prev         = MOE_CACHE_INVALID_SLOT;
        s->lru_next         = MOE_CACHE_INVALID_SLOT;
    }

    /* Allocate hash lookup table (2× slots for low load factor) */
    pool->lookup_capacity = n_slots * 2;
    if (pool->lookup_capacity < 16) {
        pool->lookup_capacity = 16;
    }
    pool->lookup_table = (int32_t *)malloc(
        (size_t)pool->lookup_capacity * sizeof(int32_t)
    );
    if (!pool->lookup_table) {
        free(pool->data_arena);
        free(pool->slots);
        pool->data_arena = NULL;
        pool->slots      = NULL;
        return AMCOLI_ERR_OUT_OF_MEMORY;
    }

    /* Fill hash table with "empty" sentinel */
    for (int32_t i = 0; i < pool->lookup_capacity; i++) {
        pool->lookup_table[i] = MOE_CACHE_INVALID_SLOT;
    }

    return AMCOLI_OK;
}

/**
 * Free a slot pool's resources.
 */
static void pool_free(struct moe_slot_pool *pool) {
    if (pool->lookup_table) { free(pool->lookup_table); pool->lookup_table = NULL; }
    if (pool->data_arena)   { free(pool->data_arena);   pool->data_arena   = NULL; }
    if (pool->slots)        { free(pool->slots);        pool->slots        = NULL; }
    pool->n_slots = 0;
    pool->n_used  = 0;
}

/**
 * Insert a key into the hash table, pointing to the given slot index.
 */
static void pool_hash_insert(
    struct moe_slot_pool *pool,
    int32_t layer_id,
    int32_t expert_id,
    int32_t slot_idx
) {
    uint64_t key = make_key(layer_id, expert_id);
    uint32_t idx = hash_key(key, pool->lookup_capacity);

    /* Linear probing */
    for (int32_t i = 0; i < pool->lookup_capacity; i++) {
        uint32_t probe = (idx + (uint32_t)i) % (uint32_t)pool->lookup_capacity;
        if (pool->lookup_table[probe] == MOE_CACHE_INVALID_SLOT) {
            pool->lookup_table[probe] = slot_idx;
            return;
        }
    }

    /* Should never happen if load factor < 1 */
    assert(false && "Hash table full — should not happen");
}

/**
 * Remove a key from the hash table.
 */
static void pool_hash_remove(
    struct moe_slot_pool *pool,
    int32_t layer_id,
    int32_t expert_id
) {
    uint64_t key = make_key(layer_id, expert_id);
    uint32_t idx = hash_key(key, pool->lookup_capacity);

    for (int32_t i = 0; i < pool->lookup_capacity; i++) {
        uint32_t probe = (idx + (uint32_t)i) % (uint32_t)pool->lookup_capacity;
        int32_t  si    = pool->lookup_table[probe];

        if (si == MOE_CACHE_INVALID_SLOT) {
            return; /* Not found */
        }

        struct moe_cache_slot *s = &pool->slots[si];
        if (s->layer_id == layer_id && s->expert_id == expert_id) {
            /* Found — remove with backward-shift deletion */
            pool->lookup_table[probe] = MOE_CACHE_INVALID_SLOT;

            /* Re-insert any displaced entries after this position */
            uint32_t next = (probe + 1) % (uint32_t)pool->lookup_capacity;
            while (pool->lookup_table[next] != MOE_CACHE_INVALID_SLOT) {
                int32_t displaced_si = pool->lookup_table[next];
                pool->lookup_table[next] = MOE_CACHE_INVALID_SLOT;

                struct moe_cache_slot *ds = &pool->slots[displaced_si];
                pool_hash_insert(pool, ds->layer_id, ds->expert_id, displaced_si);

                next = (next + 1) % (uint32_t)pool->lookup_capacity;
            }
            return;
        }
    }
}

/**
 * Look up (layer_id, expert_id) in the hash table.
 * Returns the slot index, or MOE_CACHE_INVALID_SLOT if not found.
 */
static int32_t pool_hash_lookup(
    const struct moe_slot_pool *pool,
    int32_t layer_id,
    int32_t expert_id
) {
    if (!pool->lookup_table || pool->n_slots <= 0) {
        return MOE_CACHE_INVALID_SLOT;
    }

    uint64_t key = make_key(layer_id, expert_id);
    uint32_t idx = hash_key(key, pool->lookup_capacity);

    for (int32_t i = 0; i < pool->lookup_capacity; i++) {
        uint32_t probe = (idx + (uint32_t)i) % (uint32_t)pool->lookup_capacity;
        int32_t  si    = pool->lookup_table[probe];

        if (si == MOE_CACHE_INVALID_SLOT) {
            return MOE_CACHE_INVALID_SLOT; /* Definitely not present */
        }

        const struct moe_cache_slot *s = &pool->slots[si];
        if (s->layer_id == layer_id && s->expert_id == expert_id) {
            return si;
        }
    }

    return MOE_CACHE_INVALID_SLOT;
}

/* ── LRU list manipulation ───────────────────────────────────────────── */

/**
 * Remove a slot from the LRU doubly-linked list.
 */
static void lru_remove(struct moe_slot_pool *pool, int32_t slot_idx) {
    struct moe_cache_slot *s = &pool->slots[slot_idx];
    int32_t prev = s->lru_prev;
    int32_t next = s->lru_next;

    if (prev != MOE_CACHE_INVALID_SLOT) {
        pool->slots[prev].lru_next = next;
    } else {
        pool->lru_head = next;
    }

    if (next != MOE_CACHE_INVALID_SLOT) {
        pool->slots[next].lru_prev = prev;
    } else {
        pool->lru_tail = prev;
    }

    s->lru_prev = MOE_CACHE_INVALID_SLOT;
    s->lru_next = MOE_CACHE_INVALID_SLOT;
}

/**
 * Push a slot to the head (most recently used) of the LRU list.
 */
static void lru_push_front(struct moe_slot_pool *pool, int32_t slot_idx) {
    struct moe_cache_slot *s = &pool->slots[slot_idx];
    s->lru_prev = MOE_CACHE_INVALID_SLOT;
    s->lru_next = pool->lru_head;

    if (pool->lru_head != MOE_CACHE_INVALID_SLOT) {
        pool->slots[pool->lru_head].lru_prev = slot_idx;
    }
    pool->lru_head = slot_idx;

    if (pool->lru_tail == MOE_CACHE_INVALID_SLOT) {
        pool->lru_tail = slot_idx;
    }
}

/**
 * Move a slot to the head (mark as most recently used).
 */
static void lru_touch(struct moe_slot_pool *pool, int32_t slot_idx) {
    if (pool->lru_head == slot_idx) {
        return; /* Already at head */
    }
    lru_remove(pool, slot_idx);
    lru_push_front(pool, slot_idx);
}

/* ── Find eviction victim ────────────────────────────────────────────── */

/**
 * Find the slot to evict based on the eviction policy.
 * For LRU: returns the tail (least recently used).
 * For LFU: scans for the lowest access_count.
 */
static int32_t find_eviction_victim(
    struct moe_slot_pool *pool,
    int32_t eviction_policy
) {
    if (pool->n_used <= 0) {
        return MOE_CACHE_INVALID_SLOT;
    }

    if (eviction_policy == AMCOLI_EVICT_LRU || eviction_policy == AMCOLI_EVICT_SLRU) {
        /* LRU: tail of the linked list is the victim */
        return pool->lru_tail;
    }

    if (eviction_policy == AMCOLI_EVICT_LFU) {
        /* LFU: scan all slots for minimum access_count */
        int32_t victim = MOE_CACHE_INVALID_SLOT;
        int64_t min_count = INT64_MAX;

        for (int32_t i = 0; i < pool->n_slots; i++) {
            if (pool->slots[i].state == MOE_SLOT_READY &&
                pool->slots[i].access_count < min_count)
            {
                min_count = pool->slots[i].access_count;
                victim = i;
            }
        }
        return victim;
    }

    /* Fallback: LRU */
    return pool->lru_tail;
}

/**
 * Find a free (empty) slot in the pool. Returns MOE_CACHE_INVALID_SLOT if full.
 */
static int32_t find_free_slot(const struct moe_slot_pool *pool) {
    for (int32_t i = 0; i < pool->n_slots; i++) {
        if (pool->slots[i].state == MOE_SLOT_EMPTY) {
            return i;
        }
    }
    return MOE_CACHE_INVALID_SLOT;
}

/* ── Public API ──────────────────────────────────────────────────────── */

int32_t moe_cache_init(
    struct moe_cache *cache,
    const struct amcoli_cache_params *params,
    size_t  expert_bytes,
    int32_t n_moe_layers
) {
    if (!cache || !params || expert_bytes == 0) {
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    memset(cache, 0, sizeof(*cache));
    cache->eviction_policy = params->eviction_policy;
    cache->n_layers = n_moe_layers;

    /* Compute slot counts if auto */
    int32_t vram_slots = params->vram_slots;
    int32_t ram_slots  = params->ram_slots;

    if (vram_slots == 0 && params->vram_budget_bytes > 0) {
        vram_slots = (int32_t)(params->vram_budget_bytes / (int64_t)expert_bytes);
    }
    if (ram_slots == 0 && params->ram_budget_bytes > 0) {
        ram_slots = (int32_t)(params->ram_budget_bytes / (int64_t)expert_bytes);
    }

    /* Auto-detect: use sensible defaults (will be replaced with real detection) */
    if (vram_slots == 0 && params->vram_budget_bytes == -1) {
        vram_slots = 0; /* No VRAM tier unless explicitly configured */
    }
    if (ram_slots == 0 && params->ram_budget_bytes == -1) {
        /* Default: allocate enough for 2× top-k experts per layer
         * (a reasonable starting point before auto-detection is implemented) */
        ram_slots = 32;
    }

    /* Initialize pools */
    int32_t err;
    err = pool_init(&cache->vram_pool, MOE_TIER_VRAM, vram_slots, expert_bytes);
    if (err != AMCOLI_OK) return err;

    err = pool_init(&cache->ram_pool, MOE_TIER_RAM, ram_slots, expert_bytes);
    if (err != AMCOLI_OK) {
        pool_free(&cache->vram_pool);
        return err;
    }

    /* Clear per-layer stats */
    for (int32_t i = 0; i < MOE_CACHE_MAX_LAYERS; i++) {
        cache->layer_stats[i].hits   = 0;
        cache->layer_stats[i].misses = 0;
    }

    return AMCOLI_OK;
}

void moe_cache_free(struct moe_cache *cache) {
    if (!cache) return;
    pool_free(&cache->vram_pool);
    pool_free(&cache->ram_pool);
    memset(cache, 0, sizeof(*cache));
}

bool moe_cache_lookup(
    struct moe_cache *cache,
    int32_t  layer_id,
    int32_t  expert_id,
    int32_t *tier_out,
    void   **data_out,
    size_t  *size_out
) {
    if (!cache) return false;

    /* Check VRAM first (faster tier) */
    int32_t si = pool_hash_lookup(&cache->vram_pool, layer_id, expert_id);
    if (si != MOE_CACHE_INVALID_SLOT) {
        struct moe_cache_slot *s = &cache->vram_pool.slots[si];
        if (s->state == MOE_SLOT_READY) {
            /* Update access tracking */
            cache->vram_pool.tick++;
            s->last_access_tick = cache->vram_pool.tick;
            s->access_count++;
            lru_touch(&cache->vram_pool, si);

            cache->vram_pool.stat_hits++;
            if (layer_id >= 0 && layer_id < MOE_CACHE_MAX_LAYERS) {
                cache->layer_stats[layer_id].hits++;
            }

            if (tier_out) *tier_out = MOE_TIER_VRAM;
            if (data_out) *data_out = s->data;
            if (size_out) *size_out = s->data_size;
            return true;
        }
    }

    /* Check RAM */
    si = pool_hash_lookup(&cache->ram_pool, layer_id, expert_id);
    if (si != MOE_CACHE_INVALID_SLOT) {
        struct moe_cache_slot *s = &cache->ram_pool.slots[si];
        if (s->state == MOE_SLOT_READY) {
            cache->ram_pool.tick++;
            s->last_access_tick = cache->ram_pool.tick;
            s->access_count++;
            lru_touch(&cache->ram_pool, si);

            cache->ram_pool.stat_hits++;
            if (layer_id >= 0 && layer_id < MOE_CACHE_MAX_LAYERS) {
                cache->layer_stats[layer_id].hits++;
            }

            if (tier_out) *tier_out = MOE_TIER_RAM;
            if (data_out) *data_out = s->data;
            if (size_out) *size_out = s->data_size;
            return true;
        }
    }

    /* Miss */
    cache->ram_pool.stat_misses++;
    if (layer_id >= 0 && layer_id < MOE_CACHE_MAX_LAYERS) {
        cache->layer_stats[layer_id].misses++;
    }

    return false;
}

int32_t moe_cache_insert(
    struct moe_cache *cache,
    int32_t  tier,
    int32_t  layer_id,
    int32_t  expert_id,
    const void *src_data,
    size_t   src_size,
    void   **data_out
) {
    if (!cache || !src_data) return AMCOLI_ERR_INVALID_PARAMS;

    struct moe_slot_pool *pool =
        (tier == MOE_TIER_VRAM) ? &cache->vram_pool : &cache->ram_pool;

    if (pool->n_slots <= 0) {
        return AMCOLI_ERR_INVALID_PARAMS; /* Tier disabled */
    }

    /* Check if already cached */
    int32_t existing = pool_hash_lookup(pool, layer_id, expert_id);
    if (existing != MOE_CACHE_INVALID_SLOT) {
        struct moe_cache_slot *s = &pool->slots[existing];
        if (s->state == MOE_SLOT_READY) {
            lru_touch(pool, existing);
            if (data_out) *data_out = s->data;
            return AMCOLI_OK;
        }
    }

    /* Find a free slot, or evict */
    int32_t slot_idx = find_free_slot(pool);
    if (slot_idx == MOE_CACHE_INVALID_SLOT) {
        /* Must evict */
        slot_idx = find_eviction_victim(pool, cache->eviction_policy);
        if (slot_idx == MOE_CACHE_INVALID_SLOT) {
            return AMCOLI_ERR_OUT_OF_MEMORY;
        }

        /* Evict the victim */
        struct moe_cache_slot *victim = &pool->slots[slot_idx];
        pool_hash_remove(pool, victim->layer_id, victim->expert_id);
        lru_remove(pool, slot_idx);

        victim->state    = MOE_SLOT_EMPTY;
        victim->layer_id = -1;
        victim->expert_id = -1;
        pool->n_used--;
        pool->stat_evictions++;
    }

    /* Fill the slot */
    struct moe_cache_slot *s = &pool->slots[slot_idx];
    s->layer_id         = layer_id;
    s->expert_id        = expert_id;
    s->state            = MOE_SLOT_LOADING;
    s->access_count     = 1;
    pool->tick++;
    s->last_access_tick = pool->tick;

    /* Copy data into the slot's arena region */
    size_t copy_size = (src_size <= s->data_size) ? src_size : s->data_size;
    memcpy(s->data, src_data, copy_size);

    s->state = MOE_SLOT_READY;

    /* Register in hash table and LRU list */
    pool_hash_insert(pool, layer_id, expert_id, slot_idx);
    lru_push_front(pool, slot_idx);
    pool->n_used++;

    if (data_out) *data_out = s->data;
    return AMCOLI_OK;
}

int32_t moe_cache_evict(struct moe_cache *cache, int32_t tier) {
    if (!cache) return MOE_CACHE_INVALID_SLOT;

    struct moe_slot_pool *pool =
        (tier == MOE_TIER_VRAM) ? &cache->vram_pool : &cache->ram_pool;

    int32_t victim = find_eviction_victim(pool, cache->eviction_policy);
    if (victim == MOE_CACHE_INVALID_SLOT) {
        return MOE_CACHE_INVALID_SLOT;
    }

    struct moe_cache_slot *s = &pool->slots[victim];
    pool_hash_remove(pool, s->layer_id, s->expert_id);
    lru_remove(pool, victim);

    s->state     = MOE_SLOT_EMPTY;
    s->layer_id  = -1;
    s->expert_id = -1;
    s->access_count     = 0;
    s->last_access_tick = 0;
    pool->n_used--;
    pool->stat_evictions++;

    return victim;
}

double moe_cache_hit_rate(const struct moe_cache *cache) {
    if (!cache) return 0.0;

    int64_t total_hits =
        cache->vram_pool.stat_hits + cache->ram_pool.stat_hits;
    int64_t total_misses = cache->ram_pool.stat_misses;
    int64_t total = total_hits + total_misses;

    if (total == 0) return 0.0;
    return (double)total_hits / (double)total;
}

void moe_cache_print_stats(const struct moe_cache *cache) {
    if (!cache) return;

    fprintf(stderr, "\n=== AMcoli Expert Cache Statistics ===\n");

    fprintf(stderr, "  VRAM tier: %d/%d slots used, %lld hits, %lld evictions\n",
        cache->vram_pool.n_used, cache->vram_pool.n_slots,
        (long long)cache->vram_pool.stat_hits,
        (long long)cache->vram_pool.stat_evictions);

    fprintf(stderr, "  RAM  tier: %d/%d slots used, %lld hits, %lld evictions\n",
        cache->ram_pool.n_used, cache->ram_pool.n_slots,
        (long long)cache->ram_pool.stat_hits,
        (long long)cache->ram_pool.stat_evictions);

    fprintf(stderr, "  Total misses: %lld\n",
        (long long)cache->ram_pool.stat_misses);

    fprintf(stderr, "  Overall hit rate: %.1f%%\n",
        moe_cache_hit_rate(cache) * 100.0);

    /* Per-layer breakdown (only layers with activity) */
    fprintf(stderr, "  Per-layer breakdown:\n");
    for (int32_t i = 0; i < cache->n_layers && i < MOE_CACHE_MAX_LAYERS; i++) {
        int64_t h = cache->layer_stats[i].hits;
        int64_t m = cache->layer_stats[i].misses;
        if (h + m > 0) {
            double rate = (double)h / (double)(h + m) * 100.0;
            fprintf(stderr, "    Layer %3d: %6lld hits, %6lld misses (%.1f%%)\n",
                i, (long long)h, (long long)m, rate);
        }
    }

    fprintf(stderr, "=====================================\n\n");
}
