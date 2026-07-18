/**
 * @file test-moe-cache.cpp
 * @brief Unit tests for the two-tier expert cache.
 *
 * Tests:
 *   1. Basic init and free
 *   2. Insert and lookup (cache hit)
 *   3. Cache miss
 *   4. LRU eviction ordering
 *   5. Multiple inserts with eviction
 *   6. Cache hit rate computation
 *   7. Per-layer statistics
 *   8. LFU eviction policy
 *   9. Re-insert existing expert (dedup)
 *  10. Stress test with many experts
 */

#include "llama-moe-cache.h"
#include "amcoli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_START(name) \
    fprintf(stderr, "  [TEST] %-50s ", name)

#define TEST_PASS() do { \
    fprintf(stderr, "PASS\n"); \
    tests_passed++; \
} while(0)

#define TEST_FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { TEST_FAIL(msg); return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { TEST_FAIL(msg); return; } \
} while(0)

/* ── Test helpers ────────────────────────────────────────────────────── */

static const size_t EXPERT_SIZE = 1024; /* 1 KB fake experts for testing */

static void fill_expert_data(void *buf, size_t size, int32_t layer, int32_t expert) {
    /* Fill with a recognizable pattern based on layer/expert */
    uint8_t *p = (uint8_t *)buf;
    uint8_t val = (uint8_t)((layer * 37 + expert * 13) & 0xFF);
    memset(p, val, size);
}

static bool verify_expert_data(const void *buf, size_t size, int32_t layer, int32_t expert) {
    const uint8_t *p = (const uint8_t *)buf;
    uint8_t val = (uint8_t)((layer * 37 + expert * 13) & 0xFF);
    for (size_t i = 0; i < size; i++) {
        if (p[i] != val) return false;
    }
    return true;
}

/* ── Tests ───────────────────────────────────────────────────────────── */

static void test_init_and_free(void) {
    TEST_START("init_and_free");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 4;
    params.vram_slots = 0;

    int32_t err = moe_cache_init(&cache, &params, EXPERT_SIZE, 8);
    ASSERT_EQ(err, AMCOLI_OK, "init should succeed");
    ASSERT_EQ(cache.ram_pool.n_slots, 4, "should have 4 RAM slots");
    ASSERT_EQ(cache.ram_pool.n_used, 0, "should start empty");
    ASSERT_EQ(cache.vram_pool.n_slots, 0, "VRAM should be disabled");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_insert_and_lookup(void) {
    TEST_START("insert_and_lookup");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 4;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    /* Prepare fake expert data */
    uint8_t data[EXPERT_SIZE];
    fill_expert_data(data, EXPERT_SIZE, 2, 5);

    /* Insert */
    void *cached = NULL;
    int32_t err = moe_cache_insert(&cache, MOE_TIER_RAM, 2, 5, data, EXPERT_SIZE, &cached);
    ASSERT_EQ(err, AMCOLI_OK, "insert should succeed");
    ASSERT_TRUE(cached != NULL, "should get back a pointer");
    ASSERT_TRUE(verify_expert_data(cached, EXPERT_SIZE, 2, 5), "data should match");

    /* Lookup */
    int32_t tier;
    void *found = NULL;
    size_t found_size;
    bool hit = moe_cache_lookup(&cache, 2, 5, &tier, &found, &found_size);
    ASSERT_TRUE(hit, "lookup should hit");
    ASSERT_EQ(tier, MOE_TIER_RAM, "should be in RAM");
    ASSERT_TRUE(verify_expert_data(found, found_size, 2, 5), "data should match after lookup");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_cache_miss(void) {
    TEST_START("cache_miss");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 4;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    int32_t tier;
    void *found = NULL;
    size_t found_size;
    bool hit = moe_cache_lookup(&cache, 3, 7, &tier, &found, &found_size);
    ASSERT_TRUE(!hit, "empty cache should miss");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_lru_eviction(void) {
    TEST_START("lru_eviction_order");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 3; /* Only 3 slots */
    params.eviction_policy = AMCOLI_EVICT_LRU;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    /* Insert 3 experts (fills the cache) */
    uint8_t data[EXPERT_SIZE];
    void *out;
    for (int i = 0; i < 3; i++) {
        fill_expert_data(data, EXPERT_SIZE, 0, i);
        moe_cache_insert(&cache, MOE_TIER_RAM, 0, i, data, EXPERT_SIZE, &out);
    }
    ASSERT_EQ(cache.ram_pool.n_used, 3, "cache should be full");

    /* Insert a 4th expert — should evict expert 0 (oldest) */
    fill_expert_data(data, EXPERT_SIZE, 0, 3);
    moe_cache_insert(&cache, MOE_TIER_RAM, 0, 3, data, EXPERT_SIZE, &out);

    /* Expert 0 should be evicted */
    int32_t tier;
    void *found;
    size_t sz;
    bool hit0 = moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    ASSERT_TRUE(!hit0, "expert 0 should be evicted");

    /* Experts 1, 2, 3 should still be present */
    bool hit1 = moe_cache_lookup(&cache, 0, 1, &tier, &found, &sz);
    bool hit2 = moe_cache_lookup(&cache, 0, 2, &tier, &found, &sz);
    bool hit3 = moe_cache_lookup(&cache, 0, 3, &tier, &found, &sz);
    ASSERT_TRUE(hit1 && hit2 && hit3, "experts 1,2,3 should remain");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_lru_touch_reorder(void) {
    TEST_START("lru_touch_reorder");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 3;
    params.eviction_policy = AMCOLI_EVICT_LRU;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    /* Insert experts 0, 1, 2 */
    uint8_t data[EXPERT_SIZE];
    void *out;
    for (int i = 0; i < 3; i++) {
        fill_expert_data(data, EXPERT_SIZE, 0, i);
        moe_cache_insert(&cache, MOE_TIER_RAM, 0, i, data, EXPERT_SIZE, &out);
    }

    /* Touch expert 0 (moves it to front of LRU) */
    int32_t tier;
    void *found;
    size_t sz;
    moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);

    /* Insert expert 3 — should evict expert 1 (now oldest) */
    fill_expert_data(data, EXPERT_SIZE, 0, 3);
    moe_cache_insert(&cache, MOE_TIER_RAM, 0, 3, data, EXPERT_SIZE, &out);

    bool hit0 = moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    bool hit1 = moe_cache_lookup(&cache, 0, 1, &tier, &found, &sz);
    ASSERT_TRUE(hit0, "expert 0 should survive (was touched)");
    ASSERT_TRUE(!hit1, "expert 1 should be evicted (oldest after touch)");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_hit_rate(void) {
    TEST_START("hit_rate_computation");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 4;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    /* Insert and lookup */
    uint8_t data[EXPERT_SIZE];
    void *out;
    fill_expert_data(data, EXPERT_SIZE, 0, 0);
    moe_cache_insert(&cache, MOE_TIER_RAM, 0, 0, data, EXPERT_SIZE, &out);

    int32_t tier;
    void *found;
    size_t sz;

    /* 3 hits */
    moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);

    /* 1 miss */
    moe_cache_lookup(&cache, 0, 1, &tier, &found, &sz);

    double rate = moe_cache_hit_rate(&cache);
    /* 3 hits / (3 hits + 1 miss) = 0.75 */
    ASSERT_TRUE(rate > 0.74 && rate < 0.76, "hit rate should be ~75%");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_layer_stats(void) {
    TEST_START("per_layer_statistics");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 8;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 4);

    uint8_t data[EXPERT_SIZE];
    void *out;

    /* Insert expert in layer 2 */
    fill_expert_data(data, EXPERT_SIZE, 2, 0);
    moe_cache_insert(&cache, MOE_TIER_RAM, 2, 0, data, EXPERT_SIZE, &out);

    int32_t tier;
    void *found;
    size_t sz;

    /* 2 hits on layer 2 */
    moe_cache_lookup(&cache, 2, 0, &tier, &found, &sz);
    moe_cache_lookup(&cache, 2, 0, &tier, &found, &sz);

    /* 1 miss on layer 3 */
    moe_cache_lookup(&cache, 3, 0, &tier, &found, &sz);

    ASSERT_EQ(cache.layer_stats[2].hits, 2, "layer 2 should have 2 hits");
    ASSERT_EQ(cache.layer_stats[3].misses, 1, "layer 3 should have 1 miss");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_lfu_eviction(void) {
    TEST_START("lfu_eviction_policy");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 3;
    params.eviction_policy = AMCOLI_EVICT_LFU;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    uint8_t data[EXPERT_SIZE];
    void *out;
    int32_t tier;
    void *found;
    size_t sz;

    /* Insert experts 0, 1, 2 */
    for (int i = 0; i < 3; i++) {
        fill_expert_data(data, EXPERT_SIZE, 0, i);
        moe_cache_insert(&cache, MOE_TIER_RAM, 0, i, data, EXPERT_SIZE, &out);
    }

    /* Access expert 0 many times (high frequency) */
    for (int i = 0; i < 10; i++) {
        moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    }

    /* Access expert 2 a few times */
    for (int i = 0; i < 3; i++) {
        moe_cache_lookup(&cache, 0, 2, &tier, &found, &sz);
    }

    /* Expert 1 has lowest frequency (1 from insert). Insert expert 3. */
    fill_expert_data(data, EXPERT_SIZE, 0, 3);
    moe_cache_insert(&cache, MOE_TIER_RAM, 0, 3, data, EXPERT_SIZE, &out);

    /* Expert 1 should be evicted (lowest frequency) */
    bool hit1 = moe_cache_lookup(&cache, 0, 1, &tier, &found, &sz);
    bool hit0 = moe_cache_lookup(&cache, 0, 0, &tier, &found, &sz);
    ASSERT_TRUE(!hit1, "expert 1 should be evicted (lowest freq)");
    ASSERT_TRUE(hit0, "expert 0 should survive (highest freq)");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_reinsert_dedup(void) {
    TEST_START("reinsert_deduplication");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 4;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 8);

    uint8_t data[EXPERT_SIZE];
    void *out;

    /* Insert same expert twice */
    fill_expert_data(data, EXPERT_SIZE, 1, 2);
    moe_cache_insert(&cache, MOE_TIER_RAM, 1, 2, data, EXPERT_SIZE, &out);
    moe_cache_insert(&cache, MOE_TIER_RAM, 1, 2, data, EXPERT_SIZE, &out);

    ASSERT_EQ(cache.ram_pool.n_used, 1, "should not duplicate");

    moe_cache_free(&cache);
    TEST_PASS();
}

static void test_stress_many_experts(void) {
    TEST_START("stress_100_experts_in_8_slots");

    struct moe_cache cache;
    struct amcoli_cache_params params = amcoli_cache_params_default();
    params.ram_slots = 8;
    params.eviction_policy = AMCOLI_EVICT_LRU;

    moe_cache_init(&cache, &params, EXPERT_SIZE, 4);

    uint8_t data[EXPERT_SIZE];
    void *out;

    /* Insert 100 experts across 4 layers */
    for (int i = 0; i < 100; i++) {
        int32_t layer  = i % 4;
        int32_t expert = i / 4;
        fill_expert_data(data, EXPERT_SIZE, layer, expert);
        int32_t err = moe_cache_insert(
            &cache, MOE_TIER_RAM, layer, expert, data, EXPERT_SIZE, &out
        );
        ASSERT_EQ(err, AMCOLI_OK, "insert should always succeed");
    }

    /* Cache should have exactly 8 items */
    ASSERT_EQ(cache.ram_pool.n_used, 8, "should have exactly 8 items");

    /* The last 8 inserted should be present */
    int32_t tier;
    void *found;
    size_t sz;
    for (int i = 92; i < 100; i++) {
        int32_t layer  = i % 4;
        int32_t expert = i / 4;
        bool hit = moe_cache_lookup(&cache, layer, expert, &tier, &found, &sz);
        ASSERT_TRUE(hit, "recent expert should be in cache");
        ASSERT_TRUE(verify_expert_data(found, sz, layer, expert), "data should be correct");
    }

    moe_cache_free(&cache);
    TEST_PASS();
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "\n=== AMcoli Expert Cache Tests ===\n\n");

    test_init_and_free();
    test_insert_and_lookup();
    test_cache_miss();
    test_lru_eviction();
    test_lru_touch_reorder();
    test_hit_rate();
    test_layer_stats();
    test_lfu_eviction();
    test_reinsert_dedup();
    test_stress_many_experts();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
        tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
