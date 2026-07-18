/**
 * @file test-bench.cpp
 * @brief Unit tests for the benchmark harness.
 */

#include "amcoli-bench.h"
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

/* ── GGUF Builder Helpers ────────────────────────────────────────────── */

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
    wr_u32(p, 4); /* UINT32 */
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

static uint8_t *build_test_moe_gguf(size_t *size_out) {
    size_t buf_size = 64 * 1024;
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    wr_u32(&p, 0x46475547);
    wr_u32(&p, 3);
    wr_u64(&p, 2 * 4 * 3 + 2); /* 2 layers × 4 experts × 3 tensors + 2 gates */
    wr_u64(&p, 3);

    wr_kv_u32(&p, "llama.expert_count", 4);
    wr_kv_u32(&p, "llama.expert_used_count", 2);
    wr_kv_u32(&p, "llama.block_count", 2);

    uint64_t dims[1] = {64};
    uint64_t gate_dims[2] = {64, 4};
    uint64_t offset = 0;

    for (int l = 0; l < 2; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        wr_tensor_info(&p, name, 2, gate_dims, 0, offset);
        offset += 64 * 4 * 4;
    }

    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 4; e++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 32;
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 32;
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 32;
        }
    }

    *size_out = (size_t)(p - buf) + (size_t)offset + 32;
    return buf;
}

/* ── Tests ───────────────────────────────────────────────────────────── */

static void test_run_benchmark_success(void) {
    TEST_START("run_benchmark_success");

    size_t size;
    uint8_t *data = build_test_moe_gguf(&size);

    const char *tmp_path = "test_bench_model.gguf";
    FILE *f = fopen(tmp_path, "wb");
    ASSERT_TRUE(f != NULL, "should create temp GGUF");
    fwrite(data, 1, size, f);
    fclose(f);

    struct amcoli_params params = amcoli_params_default();
    params.disk.model_path = tmp_path;
    params.verbosity = 0; /* Quiet context creation */

    int32_t err = 0;
    struct amcoli_context *ctx = amcoli_context_create(&params, &err);
    ASSERT_EQ(err, AMCOLI_OK, "ctx creation should succeed");

    struct amcoli_bench_params bparams = amcoli_bench_params_default();
    bparams.num_tokens = 50;
    bparams.zipf_exponent = 1.2;
    bparams.quiet = true;
    bparams.output_json = "test_bench_results.json";

    err = amcoli_run_benchmark(ctx, &bparams);
    ASSERT_EQ(err, AMCOLI_OK, "benchmark should run successfully");

    /* Check if JSON file was written and contains data */
    FILE *jf = fopen("test_bench_results.json", "r");
    ASSERT_TRUE(jf != NULL, "results JSON should exist");
    char line[256];
    bool found_tokens = false;
    while (fgets(line, sizeof(line), jf)) {
        if (strstr(line, "\"num_tokens\": 50")) {
            found_tokens = true;
            break;
        }
    }
    fclose(jf);
    ASSERT_TRUE(found_tokens, "JSON should contain simulated tokens count");

    /* Clean up */
    amcoli_context_free(ctx);
    remove(tmp_path);
    remove("test_bench_results.json");
    free(data);

    TEST_PASS();
}

int main(void) {
    fprintf(stderr, "\n=== AMcoli Benchmark Harness Tests ===\n\n");
    test_run_benchmark_success();
    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
        tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
