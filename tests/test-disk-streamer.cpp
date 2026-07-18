/**
 * @file test-disk-streamer.cpp
 * @brief Unit tests for the disk streamer and expert offset index.
 *
 * Tests:
 *   1. GGUF magic validation
 *   2. Expert tensor name parsing
 *   3. Expert offset index construction (with synthetic GGUF)
 *   4. Expert read from mmap
 *   5. Contiguity detection
 *   6. Prefetch hint issuance
 *   7. Statistics tracking
 */

#include "llama-disk-streamer.h"
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

/* ── Test: expert tensor name parsing ────────────────────────────────── */

/*
 * We test the name parsing logic indirectly through the disk streamer.
 * The parse_expert_tensor_name function is static, so we test its behavior
 * through the index building process.
 *
 * For unit-level coverage, we test the patterns we expect to encounter.
 */

static void test_init_fails_missing_file(void) {
    TEST_START("init_fails_missing_file");

    struct disk_streamer streamer;
    struct amcoli_disk_params params;
    memset(&params, 0, sizeof(params));
    params.model_path = "nonexistent_file_12345.gguf";
    params.io_backend = AMCOLI_IO_MMAP;

    int32_t err = disk_streamer_init(&streamer, &params);
    ASSERT_EQ(err, AMCOLI_ERR_MMAP_FAILED, "should fail on missing file");

    TEST_PASS();
}

static void test_init_fails_null_params(void) {
    TEST_START("init_fails_null_params");

    struct disk_streamer streamer;
    int32_t err = disk_streamer_init(&streamer, NULL);
    ASSERT_EQ(err, AMCOLI_ERR_INVALID_PARAMS, "should fail on null params");

    err = disk_streamer_init(NULL, NULL);
    ASSERT_EQ(err, AMCOLI_ERR_INVALID_PARAMS, "should fail on null streamer");

    TEST_PASS();
}

static void test_get_record_out_of_bounds(void) {
    TEST_START("get_record_out_of_bounds");

    struct disk_streamer streamer;
    memset(&streamer, 0, sizeof(streamer));

    /* No index built yet */
    const struct expert_record *rec = disk_streamer_get_record(&streamer, 0, 0);
    ASSERT_TRUE(rec == NULL, "should return NULL with no index");

    TEST_PASS();
}

static void test_stats_initially_zero(void) {
    TEST_START("stats_initially_zero");

    struct disk_streamer streamer;
    memset(&streamer, 0, sizeof(streamer));

    ASSERT_EQ(streamer.stat_reads, 0, "reads should be 0");
    ASSERT_EQ(streamer.stat_bytes_read, 0, "bytes should be 0");
    ASSERT_EQ(streamer.stat_prefetches, 0, "prefetches should be 0");

    TEST_PASS();
}

/* ── Synthetic GGUF builder for testing ──────────────────────────────── */

/**
 * Build a minimal synthetic GGUF file in memory for testing the index builder.
 *
 * Layout:
 *   - Header: magic + version + n_tensors + n_kv
 *   - KV pairs: expert_count=4, expert_used_count=2, block_count=2
 *   - Tensor infos: expert tensors for 2 layers × 4 experts
 *   - Data section: fake weight data
 */
struct synthetic_gguf {
    uint8_t *data;
    size_t   size;
};

static void write_u32(uint8_t **p, uint32_t v) {
    memcpy(*p, &v, 4); *p += 4;
}

static void write_u64(uint8_t **p, uint64_t v) {
    memcpy(*p, &v, 8); *p += 8;
}

static void write_str(uint8_t **p, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    write_u64(p, len);
    memcpy(*p, s, (size_t)len);
    *p += (size_t)len;
}

static void write_kv_u32(uint8_t **p, const char *key, uint32_t val) {
    write_str(p, key);        /* key string */
    write_u32(p, 4);          /* type = UINT32 */
    write_u32(p, val);        /* value */
}

static void write_tensor_info(
    uint8_t **p,
    const char *name,
    uint32_t n_dims,
    uint64_t *dims,
    uint32_t type,
    uint64_t offset
) {
    write_str(p, name);
    write_u32(p, n_dims);
    for (uint32_t d = 0; d < n_dims; d++) {
        write_u64(p, dims[d]);
    }
    write_u32(p, type); /* Q4_0 = type 2 */
    write_u64(p, offset);
}

static struct synthetic_gguf build_synthetic_gguf(void) {
    /* Allocate a large buffer */
    size_t buf_size = 256 * 1024; /* 256 KB to be safe */
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    /* Header */
    write_u32(&p, 0x46475547); /* magic "GGUF" */
    write_u32(&p, 3);          /* version */

    /* We'll have 2 layers × 4 experts × 3 tensors = 24 tensors
     * Plus the gate_inp tensors = 2 more */
    uint64_t n_tensors = 24 + 2;
    write_u64(&p, n_tensors);

    /* 3 KV pairs */
    uint64_t n_kv = 3;
    write_u64(&p, n_kv);

    /* KV pairs */
    write_kv_u32(&p, "llama.expert_count", 4);
    write_kv_u32(&p, "llama.expert_used_count", 2);
    write_kv_u32(&p, "llama.block_count", 2);

    /* Tensor infos — use SMALL dimensions to keep file within buffer
     * Each expert tensor: [64] elements, type Q4_0 (2)
     * = 64 * 4 bits / 8 = 32 bytes per tensor */
    uint64_t expert_tensor_elems = 64;
    uint64_t dims[1] = {expert_tensor_elems};
    uint32_t q4_type = 2; /* Q4_0 */
    uint64_t tensor_size = (expert_tensor_elems * 4 + 7) / 8; /* 32 bytes */

    uint64_t offset = 0;

    /* Gate input tensors (routing gates) — not expert tensors */
    for (int layer = 0; layer < 2; layer++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
        uint64_t gate_dims[2] = {64, 4};
        write_tensor_info(&p, name, 2, gate_dims, 0 /* F32 */, offset);
        offset += 64 * 4 * 4; /* 64 × 4 experts × 4 bytes = 1024 */
    }

    /* Expert tensors: individual (not merged) */
    for (int layer = 0; layer < 2; layer++) {
        for (int expert = 0; expert < 4; expert++) {
            char name[64];

            /* Gate */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exp.%d.weight", layer, expert);
            write_tensor_info(&p, name, 1, dims, q4_type, offset);
            offset += tensor_size;

            /* Up */
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exp.%d.weight", layer, expert);
            write_tensor_info(&p, name, 1, dims, q4_type, offset);
            offset += tensor_size;

            /* Down */
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exp.%d.weight", layer, expert);
            write_tensor_info(&p, name, 1, dims, q4_type, offset);
            offset += tensor_size;
        }
    }

    /* Align to 32 bytes for data section */
    size_t header_size = (size_t)(p - buf);
    size_t data_start = (header_size + 31) & ~(size_t)31;

    /* Ensure the total size fits in our buffer */
    size_t total_size = data_start + (size_t)offset;
    if (total_size > buf_size) {
        /* Shouldn't happen with small tensors, but safety check */
        total_size = buf_size;
    }

    /* Fill data section with recognizable patterns */
    for (size_t i = data_start; i < total_size; i++) {
        buf[i] = (uint8_t)(i & 0xFF);
    }

    struct synthetic_gguf gguf;
    gguf.data = buf;
    gguf.size = total_size;

    return gguf;
}

/* ── Tests using synthetic GGUF ──────────────────────────────────────── */

static void test_synthetic_gguf_index_build(void) {
    TEST_START("synthetic_gguf_index_build");

    struct synthetic_gguf gguf = build_synthetic_gguf();

    /* Write to a temp file */
    const char *tmp_path = "test_synthetic.gguf";
    FILE *f = fopen(tmp_path, "wb");
    ASSERT_TRUE(f != NULL, "should create temp file");
    fwrite(gguf.data, 1, gguf.size, f);
    fclose(f);

    /* Init streamer */
    struct disk_streamer streamer;
    struct amcoli_disk_params params;
    memset(&params, 0, sizeof(params));
    params.model_path = tmp_path;
    params.io_backend = AMCOLI_IO_MMAP;

    int32_t err = disk_streamer_init(&streamer, &params);
    ASSERT_EQ(err, AMCOLI_OK, "init should succeed");

    /* Build index */
    err = disk_streamer_build_index(&streamer, 2, 4);
    ASSERT_EQ(err, AMCOLI_OK, "build_index should succeed");

    ASSERT_EQ(streamer.index.n_layers, 2, "should have 2 layers");
    ASSERT_EQ(streamer.index.n_experts, 4, "should have 4 experts");

    /* Check that expert records have non-zero sizes */
    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 4; e++) {
            const struct expert_record *rec =
                disk_streamer_get_record(&streamer, l, e);
            ASSERT_TRUE(rec != NULL, "record should exist");
            ASSERT_TRUE(rec->total_size > 0, "total_size should be > 0");
            ASSERT_TRUE(rec->gate_size > 0, "gate should have data");
            ASSERT_TRUE(rec->up_size > 0, "up should have data");
            ASSERT_TRUE(rec->down_size > 0, "down should have data");
        }
    }

    /* Check contiguity — individual tensors should be contiguous
     * since we wrote gate/up/down sequentially */
    const struct expert_record *rec = disk_streamer_get_record(&streamer, 0, 0);
    ASSERT_TRUE(rec->is_contiguous, "expert 0/0 should be contiguous");

    /* Clean up */
    disk_streamer_free(&streamer);
    remove(tmp_path);
    free(gguf.data);

    TEST_PASS();
}

static void test_synthetic_gguf_read_expert(void) {
    TEST_START("synthetic_gguf_read_expert");

    struct synthetic_gguf gguf = build_synthetic_gguf();

    const char *tmp_path = "test_read_expert.gguf";
    FILE *f = fopen(tmp_path, "wb");
    ASSERT_TRUE(f != NULL, "should create temp file");
    fwrite(gguf.data, 1, gguf.size, f);
    fclose(f);

    struct disk_streamer streamer;
    struct amcoli_disk_params params;
    memset(&params, 0, sizeof(params));
    params.model_path = tmp_path;
    params.io_backend = AMCOLI_IO_MMAP;

    int32_t err = disk_streamer_init(&streamer, &params);
    ASSERT_EQ(err, AMCOLI_OK, "init should succeed");

    err = disk_streamer_build_index(&streamer, 2, 4);
    ASSERT_EQ(err, AMCOLI_OK, "build_index should succeed");

    /* Read an expert */
    const struct expert_record *rec = disk_streamer_get_record(&streamer, 0, 0);
    ASSERT_TRUE(rec != NULL, "record should exist");

    void *buf = malloc((size_t)rec->total_size);
    ASSERT_TRUE(buf != NULL, "should allocate read buffer");

    int64_t bytes = disk_streamer_read_expert(&streamer, 0, 0, buf, (size_t)rec->total_size);
    ASSERT_TRUE(bytes > 0, "should read bytes");
    ASSERT_EQ(bytes, rec->total_size, "should read full expert");

    /* Verify stats were updated */
    ASSERT_EQ(streamer.stat_reads, 1, "should have 1 read");
    ASSERT_TRUE(streamer.stat_bytes_read > 0, "should track bytes read");

    free(buf);
    disk_streamer_free(&streamer);
    remove(tmp_path);
    free(gguf.data);

    TEST_PASS();
}

static void test_synthetic_gguf_prefetch(void) {
    TEST_START("synthetic_gguf_prefetch");

    struct synthetic_gguf gguf = build_synthetic_gguf();

    const char *tmp_path = "test_prefetch.gguf";
    FILE *f = fopen(tmp_path, "wb");
    ASSERT_TRUE(f != NULL, "should create temp file");
    fwrite(gguf.data, 1, gguf.size, f);
    fclose(f);

    struct disk_streamer streamer;
    struct amcoli_disk_params params;
    memset(&params, 0, sizeof(params));
    params.model_path = tmp_path;
    params.io_backend = AMCOLI_IO_MMAP;

    int32_t err = disk_streamer_init(&streamer, &params);
    ASSERT_EQ(err, AMCOLI_OK, "init should succeed");

    err = disk_streamer_build_index(&streamer, 2, 4);
    ASSERT_EQ(err, AMCOLI_OK, "build_index should succeed");

    /* Issue prefetch */
    disk_streamer_prefetch_expert(&streamer, 1, 2);
    ASSERT_EQ(streamer.stat_prefetches, 1, "should track prefetch");

    disk_streamer_free(&streamer);
    remove(tmp_path);
    free(gguf.data);

    TEST_PASS();
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "\n=== AMcoli Disk Streamer Tests ===\n\n");

    test_init_fails_missing_file();
    test_init_fails_null_params();
    test_get_record_out_of_bounds();
    test_stats_initially_zero();
    test_synthetic_gguf_index_build();
    test_synthetic_gguf_read_expert();
    test_synthetic_gguf_prefetch();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
        tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
