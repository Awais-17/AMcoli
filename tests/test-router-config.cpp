/**
 * @file test-router-config.cpp
 * @brief Unit tests for MoE router configuration extraction.
 *
 * Tests:
 *   1. Non-MoE model rejection
 *   2. Basic Mixtral-style extraction (no shared experts)
 *   3. DeepSeek/GLM-style extraction (with shared experts)
 *   4. Config validation
 *   5. Budget computation
 *   6. Edge cases (missing metadata, zero experts)
 */

#include "llama-moe-router.h"
#include "amcoli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ── Synthetic GGUF builder helpers ──────────────────────────────────── */

static void wr_u32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4); *p += 4; }
static void wr_u64(uint8_t **p, uint64_t v) { memcpy(*p, &v, 8); *p += 8; }

static void wr_str(uint8_t **p, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    wr_u64(p, len);
    memcpy(*p, s, (size_t)len);
    *p += (size_t)len;
}

static void wr_kv_u32(uint8_t **p, const char *key, uint32_t val) {
    wr_str(p, key);
    wr_u32(p, 4); /* type = UINT32 */
    wr_u32(p, val);
}

static void wr_tensor_info(
    uint8_t **p, const char *name,
    uint32_t n_dims, uint64_t *dims,
    uint32_t type, uint64_t offset
) {
    wr_str(p, name);
    wr_u32(p, n_dims);
    for (uint32_t d = 0; d < n_dims; d++) wr_u64(p, dims[d]);
    wr_u32(p, type);
    wr_u64(p, offset);
}

/**
 * Build a Mixtral-style GGUF: 2 layers, 8 experts, top-2, no shared.
 */
static uint8_t *build_mixtral_gguf(size_t *size_out) {
    size_t buf_size = 64 * 1024;
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    wr_u32(&p, 0x46475547); /* magic */
    wr_u32(&p, 3);          /* version */

    uint64_t n_tensors = 2 * 8 * 3 + 2; /* 2 layers × 8 experts × 3 tensors + 2 gates */
    wr_u64(&p, n_tensors);
    wr_u64(&p, 3); /* n_kv */

    wr_kv_u32(&p, "llama.expert_count", 8);
    wr_kv_u32(&p, "llama.expert_used_count", 2);
    wr_kv_u32(&p, "llama.block_count", 2);

    uint64_t dims[1] = {2048};
    uint64_t gate_dims[2] = {2048, 8};
    uint64_t offset = 0;

    /* Gate inputs */
    for (int l = 0; l < 2; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        wr_tensor_info(&p, name, 2, gate_dims, 0, offset);
        offset += 2048 * 8 * 4;
    }

    /* Individual expert tensors */
    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 8; e++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 1024;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 1024;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 1024;
        }
    }

    *size_out = (size_t)(p - buf) + (size_t)offset + 32;
    return buf;
}

/**
 * Build a DeepSeek-style GGUF: 2 layers, 16 experts, top-4, with shared experts.
 */
static uint8_t *build_deepseek_gguf(size_t *size_out) {
    size_t buf_size = 128 * 1024;
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    wr_u32(&p, 0x46475547);
    wr_u32(&p, 3);

    uint64_t n_tensors = 2 * 16 * 3 + 2 * 3 + 2;
    /* 2 layers × 16 experts × 3 + 2 layers × 3 shared + 2 gates */
    wr_u64(&p, n_tensors);
    wr_u64(&p, 3);

    wr_kv_u32(&p, "deepseek.expert_count", 16);
    wr_kv_u32(&p, "deepseek.expert_used_count", 4);
    wr_kv_u32(&p, "deepseek.block_count", 2);

    uint64_t dims[1] = {1024};
    uint64_t gate_dims[2] = {1024, 16};
    uint64_t offset = 0;

    /* Gate inputs */
    for (int l = 0; l < 2; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        wr_tensor_info(&p, name, 2, gate_dims, 0, offset);
        offset += 1024 * 16 * 4;
    }

    /* Shared expert tensors */
    for (int l = 0; l < 2; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
        wr_tensor_info(&p, name, 1, dims, 0, offset);
        offset += 1024 * 4;

        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
        wr_tensor_info(&p, name, 1, dims, 0, offset);
        offset += 1024 * 4;

        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
        wr_tensor_info(&p, name, 1, dims, 0, offset);
        offset += 1024 * 4;
    }

    /* Routed expert tensors (individual) */
    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 16; e++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 512;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 512;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 512;
        }
    }

    *size_out = (size_t)(p - buf) + (size_t)offset + 32;
    return buf;
}

/**
 * Build a non-MoE GGUF (no expert_count).
 */
static uint8_t *build_dense_gguf(size_t *size_out) {
    size_t buf_size = 4096;
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    wr_u32(&p, 0x46475547);
    wr_u32(&p, 3);
    wr_u64(&p, 1); /* 1 tensor */
    wr_u64(&p, 1); /* 1 KV */

    wr_kv_u32(&p, "llama.block_count", 32);

    /* One non-expert tensor */
    uint64_t dims[1] = {4096};
    wr_tensor_info(&p, "blk.0.attn_q.weight", 1, dims, 0, 0);

    *size_out = (size_t)(p - buf) + 4096 + 32;
    return buf;
}

/* ── Tests ───────────────────────────────────────────────────────────── */

static void test_dense_model_rejected(void) {
    TEST_START("dense_model_rejected");

    size_t size;
    uint8_t *data = build_dense_gguf(&size);

    struct amcoli_moe_config config;
    int32_t err = moe_router_extract_config(data, size, &config);
    ASSERT_EQ(err, AMCOLI_ERR_NOT_MOE, "dense model should be rejected");

    free(data);
    TEST_PASS();
}

static void test_mixtral_extraction(void) {
    TEST_START("mixtral_style_extraction");

    size_t size;
    uint8_t *data = build_mixtral_gguf(&size);

    struct amcoli_moe_config config;
    int32_t err = moe_router_extract_config(data, size, &config);
    ASSERT_EQ(err, AMCOLI_OK, "extraction should succeed");
    ASSERT_EQ(config.n_expert, 8, "should have 8 experts");
    ASSERT_EQ(config.n_expert_used, 2, "should be top-2");
    ASSERT_EQ(config.n_total_layers, 2, "should have 2 layers");
    ASSERT_EQ(config.n_moe_layers, 2, "all layers should be MoE");
    ASSERT_TRUE(!config.has_shared_experts, "Mixtral has no shared experts");
    ASSERT_TRUE(config.expert_total_bytes > 0, "should have computed expert size");

    moe_config_free(&config);
    free(data);
    TEST_PASS();
}

static void test_deepseek_extraction(void) {
    TEST_START("deepseek_style_extraction");

    size_t size;
    uint8_t *data = build_deepseek_gguf(&size);

    struct amcoli_moe_config config;
    int32_t err = moe_router_extract_config(data, size, &config);
    ASSERT_EQ(err, AMCOLI_OK, "extraction should succeed");
    ASSERT_EQ(config.n_expert, 16, "should have 16 experts");
    ASSERT_EQ(config.n_expert_used, 4, "should be top-4");
    ASSERT_EQ(config.n_total_layers, 2, "should have 2 layers");
    ASSERT_TRUE(config.has_shared_experts, "DeepSeek should have shared experts");
    ASSERT_TRUE(config.resident_bytes > 0, "shared experts add to resident bytes");
    ASSERT_TRUE(config.routed_bytes > 0, "should have routed bytes");

    moe_config_free(&config);
    free(data);
    TEST_PASS();
}

static void test_config_validation(void) {
    TEST_START("config_validation");

    struct amcoli_moe_config config;
    memset(&config, 0, sizeof(config));

    /* Invalid: zero experts */
    config.n_expert = 0;
    ASSERT_EQ(moe_config_validate(&config), AMCOLI_ERR_INVALID_PARAMS,
        "zero experts should fail");

    /* Invalid: top-k > n_expert */
    config.n_expert = 4;
    config.n_expert_used = 5;
    config.n_moe_layers = 1;
    ASSERT_EQ(moe_config_validate(&config), AMCOLI_ERR_INVALID_PARAMS,
        "top-k > n_expert should fail");

    /* Valid */
    config.n_expert = 8;
    config.n_expert_used = 2;
    config.n_moe_layers = 4;
    config.n_total_layers = 32;
    config.expert_total_bytes = 1024 * 1024;
    ASSERT_EQ(moe_config_validate(&config), AMCOLI_OK, "valid config should pass");

    TEST_PASS();
}

static void test_budget_computation(void) {
    TEST_START("budget_computation");

    struct amcoli_moe_config config;
    memset(&config, 0, sizeof(config));
    config.n_expert = 8;
    config.n_expert_used = 2;
    config.expert_total_bytes = 1024 * 1024; /* 1 MB per expert */
    config.resident_bytes = 10 * 1024 * 1024; /* 10 MB resident */
    config.routed_bytes = 8 * 1024 * 1024;    /* 8 MB routed */

    int64_t vram, ram, disk;
    moe_config_compute_budget(&config, 4, 8, &vram, &ram, &disk);

    ASSERT_EQ(vram, 4 * 1024 * 1024, "VRAM should be 4 × expert_bytes");
    ASSERT_EQ(ram, 10 * 1024 * 1024 + 8 * 1024 * 1024, "RAM should be resident + 8 × expert_bytes");
    ASSERT_EQ(disk, 10 * 1024 * 1024 + 8 * 1024 * 1024, "disk should be total model");

    TEST_PASS();
}

static void test_null_params(void) {
    TEST_START("null_params_handling");

    struct amcoli_moe_config config;
    int32_t err = moe_router_extract_config(NULL, 0, &config);
    ASSERT_EQ(err, AMCOLI_ERR_INVALID_PARAMS, "null data should fail");

    err = moe_router_extract_config((void *)1, 100, NULL);
    ASSERT_EQ(err, AMCOLI_ERR_INVALID_PARAMS, "null output should fail");

    TEST_PASS();
}

static void test_print_functions(void) {
    TEST_START("print_functions_no_crash");

    /* Just ensure print functions don't crash on valid/null inputs */
    struct amcoli_moe_config config;
    memset(&config, 0, sizeof(config));
    config.n_expert = 8;
    config.n_expert_used = 2;
    config.n_total_layers = 32;
    config.n_moe_layers = 32;
    config.expert_total_bytes = 50 * 1024 * 1024;
    config.resident_bytes = 5LL * 1024 * 1024 * 1024;
    config.routed_bytes = 400LL * 1024 * 1024 * 1024;

    /* Redirect stderr to suppress output during test */
    moe_config_print(&config);
    moe_config_print(NULL); /* Should not crash */

    TEST_PASS();
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "\n=== AMcoli Router Config Tests ===\n\n");

    test_dense_model_rejected();
    test_mixtral_extraction();
    test_deepseek_extraction();
    test_config_validation();
    test_budget_computation();
    test_null_params();
    test_print_functions();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
        tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
