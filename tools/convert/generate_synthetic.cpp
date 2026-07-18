/**
 * @file generate_synthetic.cpp
 * @brief Generates a synthetic MoE GGUF model for testing and benchmarking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output.gguf> [n_layers] [n_experts]\n", argv[0]);
        return 1;
    }

    const char *out_path = argv[1];
    int32_t n_layers = argc >= 3 ? atoi(argv[2]) : 32;
    int32_t n_experts = argc >= 4 ? atoi(argv[3]) : 8;

    fprintf(stderr, "Generating synthetic GGUF MoE model: %s (%d layers, %d experts)...\n",
        out_path, n_layers, n_experts);

    size_t buf_size = 5 * 1024 * 1024; /* 5 MB buffer */
    uint8_t *buf = (uint8_t *)calloc(1, buf_size);
    uint8_t *p = buf;

    wr_u32(&p, 0x46475547); /* magic */
    wr_u32(&p, 3);          /* version */

    /* Total tensors = n_layers * n_experts * 3 + n_layers */
    uint64_t n_tensors = (uint64_t)n_layers * n_experts * 3 + n_layers;
    wr_u64(&p, n_tensors);
    wr_u64(&p, 3); /* KV count */

    wr_kv_u32(&p, "llama.expert_count", n_experts);
    wr_kv_u32(&p, "llama.expert_used_count", 2);
    wr_kv_u32(&p, "llama.block_count", n_layers);

    uint64_t dims[1] = {128}; /* 128 elements */
    uint64_t gate_dims[2] = {128, (uint64_t)n_experts};
    uint64_t offset = 0;

    /* Gate input tensors */
    for (int l = 0; l < n_layers; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        wr_tensor_info(&p, name, 2, gate_dims, 0, offset);
        offset += 128 * n_experts * 4;
    }

    /* Expert tensors (individual) */
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            char name[64];
            
            /* Gate */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 64; /* Q4 size */

            /* Up */
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 64;

            /* Down */
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exp.%d.weight", l, e);
            wr_tensor_info(&p, name, 1, dims, 2, offset);
            offset += 64;
        }
    }

    /* Align to 32 bytes for data section */
    size_t header_size = (size_t)(p - buf);
    size_t data_start = (header_size + 31) & ~31;

    /* Write to file */
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "Error: Failed to open output file %s\n", out_path);
        free(buf);
        return 1;
    }

    /* Write header */
    fwrite(buf, 1, data_start, f);

    /* Write fake data section */
    uint8_t *fake_data = (uint8_t *)calloc(1, 1024);
    size_t remaining = (size_t)offset;
    while (remaining > 0) {
        size_t chunk = remaining < 1024 ? remaining : 1024;
        fwrite(fake_data, 1, chunk, f);
        remaining -= chunk;
    }

    fclose(f);
    free(fake_data);
    free(buf);

    fprintf(stderr, "Successfully generated %s\n", out_path);
    return 0;
}
