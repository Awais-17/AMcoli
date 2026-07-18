/**
 * @file llama-moe-router.cpp
 * @brief Generic MoE router configuration extraction from GGUF metadata.
 *
 * Extracts a model-agnostic MoE configuration by parsing GGUF KV pairs
 * and scanning tensor names. No architecture-specific code paths.
 *
 * Supported metadata keys (with wildcard prefix matching):
 *   - *.expert_count         → number of routed experts per MoE layer
 *   - *.expert_used_count    → top-k experts activated per token
 *   - *.block_count          → total transformer layers
 *   - *.feed_forward_length  → FFN hidden dim (for size estimation)
 *
 * Supported tensor name patterns:
 *   Routed:  blk.{L}.ffn_{gate,up,down}_exp{s}
 *   Shared:  blk.{L}.ffn_{gate,up,down}_shexp{s}
 */

#include "llama-moe-router.h"
#include "llama-moe-cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── GGUF parsing (shared with disk-streamer, inlined for now) ───── */

#define GGUF_MAGIC_VAL 0x46475547

enum {
    KV_TYPE_UINT8    = 0,
    KV_TYPE_INT8     = 1,
    KV_TYPE_UINT16   = 2,
    KV_TYPE_INT16    = 3,
    KV_TYPE_UINT32   = 4,
    KV_TYPE_INT32    = 5,
    KV_TYPE_FLOAT32  = 6,
    KV_TYPE_BOOL     = 7,
    KV_TYPE_STRING   = 8,
    KV_TYPE_ARRAY    = 9,
    KV_TYPE_UINT64   = 10,
    KV_TYPE_INT64    = 11,
    KV_TYPE_FLOAT64  = 12,
};

struct reader {
    const uint8_t *base;
    size_t size;
    size_t pos;
};

static bool r_has(const struct reader *r, size_t n) {
    return (r->pos + n) <= r->size;
}

static uint32_t r_u32(struct reader *r) {
    if (!r_has(r, 4)) return 0;
    uint32_t v; memcpy(&v, r->base + r->pos, 4); r->pos += 4; return v;
}

static uint64_t r_u64(struct reader *r) {
    if (!r_has(r, 8)) return 0;
    uint64_t v; memcpy(&v, r->base + r->pos, 8); r->pos += 8; return v;
}

static const char *r_str(struct reader *r, uint64_t *len_out) {
    uint64_t len = r_u64(r);
    if (len_out) *len_out = len;
    if (!r_has(r, (size_t)len)) { if (len_out) *len_out = 0; return NULL; }
    const char *s = (const char *)(r->base + r->pos);
    r->pos += (size_t)len;
    return s;
}

static bool r_skip_val(struct reader *r, uint32_t type) {
    switch (type) {
        case KV_TYPE_UINT8:  case KV_TYPE_INT8:  case KV_TYPE_BOOL: r->pos += 1; break;
        case KV_TYPE_UINT16: case KV_TYPE_INT16: r->pos += 2; break;
        case KV_TYPE_UINT32: case KV_TYPE_INT32: case KV_TYPE_FLOAT32: r->pos += 4; break;
        case KV_TYPE_UINT64: case KV_TYPE_INT64: case KV_TYPE_FLOAT64: r->pos += 8; break;
        case KV_TYPE_STRING: { uint64_t l; r_str(r, &l); break; }
        case KV_TYPE_ARRAY: {
            uint32_t at = r_u32(r);
            uint64_t al = r_u64(r);
            for (uint64_t i = 0; i < al; i++) { if (!r_skip_val(r, at)) return false; }
            break;
        }
        default: return false;
    }
    return r_has(r, 0);
}

static uint32_t r_read_u32_val(struct reader *r) {
    return r_u32(r);
}

/* ── String matching helpers ─────────────────────────────────────────── */

/**
 * Check if a key string ends with the given suffix.
 * E.g., "llama.expert_count" ends with "expert_count".
 */
static bool key_ends_with(
    const char *key, size_t key_len,
    const char *suffix, size_t suffix_len
) {
    if (key_len < suffix_len) return false;
    return memcmp(key + key_len - suffix_len, suffix, suffix_len) == 0;
}

/**
 * Check if a tensor name contains a substring.
 */
static bool name_contains(
    const char *name, size_t name_len,
    const char *sub
) {
    size_t sub_len = strlen(sub);
    if (name_len < sub_len) return false;

    for (size_t i = 0; i <= name_len - sub_len; i++) {
        if (memcmp(name + i, sub, sub_len) == 0) return true;
    }
    return false;
}

/**
 * Extract the layer number from a tensor name like "blk.{L}.ffn_..."
 * Returns -1 if not found.
 */
static int32_t extract_layer_from_name(const char *name, size_t name_len) {
    if (name_len < 5 || memcmp(name, "blk.", 4) != 0) return -1;

    const char *p = name + 4;
    char *end;
    long layer = strtol(p, &end, 10);
    if (end == p || *end != '.') return -1;
    return (int32_t)layer;
}

/* ── GGUF tensor type → byte size approximation ─────────────────────── */

static const int g_type_bits[] = {
    32, 16, 4, 5, 5, 6, 8, 9, 3, 3, 5, 6, 6, 8, 2, 3, 2, 5, 4, 2, 16
};
#define N_TENSOR_TYPES 21

/* ── Main extraction function ────────────────────────────────────────── */

int32_t moe_router_extract_config(
    const void *file_data,
    size_t      file_size,
    struct amcoli_moe_config *config_out
) {
    if (!file_data || file_size < 24 || !config_out) {
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    memset(config_out, 0, sizeof(*config_out));

    struct reader r;
    r.base = (const uint8_t *)file_data;
    r.size = file_size;
    r.pos  = 0;

    /* Read GGUF header */
    uint32_t magic = r_u32(&r);
    if (magic != GGUF_MAGIC_VAL) {
        return AMCOLI_ERR_IO_FAILED;
    }

    uint32_t version   = r_u32(&r); (void)version;
    uint64_t n_tensors = r_u64(&r);
    uint64_t n_kv      = r_u64(&r);

    /* ── Phase 1: Scan KV pairs for MoE metadata ──────────────────── */

    int32_t expert_count      = 0;
    int32_t expert_used_count = 0;
    int32_t block_count       = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        /* Read key */
        uint64_t key_len;
        const char *key = r_str(&r, &key_len);
        if (!key) break;

        /* Read value type */
        uint32_t vtype = r_u32(&r);

        /* Check for known keys before skipping value */
        if (vtype == KV_TYPE_UINT32) {
            size_t saved_pos = r.pos;

            if (key_ends_with(key, (size_t)key_len, "expert_count", 12) &&
                !key_ends_with(key, (size_t)key_len, "expert_used_count", 17))
            {
                expert_count = (int32_t)r_read_u32_val(&r);
                continue;
            }

            r.pos = saved_pos;
            if (key_ends_with(key, (size_t)key_len, "expert_used_count", 17)) {
                expert_used_count = (int32_t)r_read_u32_val(&r);
                continue;
            }

            r.pos = saved_pos;
            if (key_ends_with(key, (size_t)key_len, "block_count", 11)) {
                block_count = (int32_t)r_read_u32_val(&r);
                continue;
            }

            r.pos = saved_pos;
        }

        /* Skip the value */
        if (!r_skip_val(&r, vtype)) break;
    }

    /* Check if this is actually an MoE model */
    if (expert_count <= 1) {
        return AMCOLI_ERR_NOT_MOE;
    }

    config_out->n_expert      = expert_count;
    config_out->n_expert_used = expert_used_count > 0 ? expert_used_count : 2; /* Default top-2 */
    config_out->n_total_layers = block_count;

    /* ── Phase 2: Scan tensor infos for expert patterns ───────────── */

    /* Track which layers have routed experts and shared experts */
    bool layer_has_routed[MOE_CACHE_MAX_LAYERS] = {false};
    bool layer_has_shared[MOE_CACHE_MAX_LAYERS] = {false};
    int32_t max_layer_seen = -1;

    /* Track per-expert tensor sizes (from first expert found) */
    int64_t first_gate_size = 0;
    int64_t first_up_size   = 0;
    int64_t first_down_size = 0;

    /* Total size tracking */
    int64_t total_expert_tensor_bytes = 0;
    int64_t total_non_expert_bytes    = 0;

    for (uint64_t t = 0; t < n_tensors; t++) {
        /* Read tensor name */
        uint64_t name_len;
        const char *name = r_str(&r, &name_len);
        if (!name) break;

        /* Read n_dims */
        uint32_t n_dims = r_u32(&r);

        /* Read dims and compute element count */
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = r_u64(&r);
            n_elements *= dim;
        }

        /* Read type */
        uint32_t tensor_type = r_u32(&r);

        /* Read offset (skip it, we don't need it here) */
        r_u64(&r);

        /* Compute size in bytes */
        int bits = 32;
        if (tensor_type < N_TENSOR_TYPES) bits = g_type_bits[tensor_type];
        int64_t tensor_bytes = (int64_t)((n_elements * (uint64_t)bits + 7) / 8);

        /* Classify the tensor */
        int32_t layer = extract_layer_from_name(name, (size_t)name_len);

        bool is_routed_gate = name_contains(name, (size_t)name_len, "ffn_gate_exp");
        bool is_routed_up   = name_contains(name, (size_t)name_len, "ffn_up_exp");
        bool is_routed_down = name_contains(name, (size_t)name_len, "ffn_down_exp");
        bool is_routed = is_routed_gate || is_routed_up || is_routed_down;

        bool is_shared_gate = name_contains(name, (size_t)name_len, "ffn_gate_shexp");
        bool is_shared_up   = name_contains(name, (size_t)name_len, "ffn_up_shexp");
        bool is_shared_down = name_contains(name, (size_t)name_len, "ffn_down_shexp");
        bool is_shared = is_shared_gate || is_shared_up || is_shared_down;

        if (is_routed && layer >= 0 && layer < MOE_CACHE_MAX_LAYERS) {
            layer_has_routed[layer] = true;
            if (layer > max_layer_seen) max_layer_seen = layer;

            total_expert_tensor_bytes += tensor_bytes;

            /* Capture per-expert size from first gate/up/down tensor.
             * For merged tensors (all experts), divide by expert_count. */
            if (is_routed_gate && first_gate_size == 0) {
                bool is_merged = name_contains(name, (size_t)name_len, "exps");
                first_gate_size = is_merged ?
                    tensor_bytes / expert_count : tensor_bytes;
            }
            if (is_routed_up && first_up_size == 0) {
                bool is_merged = name_contains(name, (size_t)name_len, "exps");
                first_up_size = is_merged ?
                    tensor_bytes / expert_count : tensor_bytes;
            }
            if (is_routed_down && first_down_size == 0) {
                bool is_merged = name_contains(name, (size_t)name_len, "exps");
                first_down_size = is_merged ?
                    tensor_bytes / expert_count : tensor_bytes;
            }
        } else if (is_shared && layer >= 0 && layer < MOE_CACHE_MAX_LAYERS) {
            layer_has_shared[layer] = true;
            if (layer > max_layer_seen) max_layer_seen = layer;

            /* Shared expert bytes counted as resident (always in RAM) */
            total_non_expert_bytes += tensor_bytes;
        } else {
            /* Non-expert tensor (attention, embeddings, norms, etc.) */
            total_non_expert_bytes += tensor_bytes;
        }
    }

    /* ── Phase 3: Build the configuration ─────────────────────────── */

    /* Count MoE layers and detect shared experts */
    int32_t n_moe_layers = 0;
    bool has_shared = false;
    int32_t n_shared_count = 0;

    for (int32_t l = 0; l <= max_layer_seen && l < MOE_CACHE_MAX_LAYERS; l++) {
        if (layer_has_routed[l]) n_moe_layers++;
        if (layer_has_shared[l]) {
            has_shared = true;
            n_shared_count++;
        }
    }

    /* If no MoE layers found from tensors, fallback: assume all layers are MoE
     * (this handles cases where tensor names don't match our patterns) */
    if (n_moe_layers == 0 && expert_count > 1) {
        n_moe_layers = block_count > 0 ? block_count : 1;
        for (int32_t l = 0; l < n_moe_layers; l++) {
            layer_has_routed[l] = true;
        }
        max_layer_seen = n_moe_layers - 1;
    }

    config_out->n_moe_layers     = n_moe_layers;
    config_out->has_shared_experts = has_shared;
    config_out->n_shared_expert   = n_shared_count > 0 ? 1 : 0; /* Per-layer shared group */

    /* Per-expert byte sizes */
    config_out->expert_bytes_gate = first_gate_size;
    config_out->expert_bytes_up   = first_up_size;
    config_out->expert_bytes_down = first_down_size;
    config_out->expert_total_bytes = first_gate_size + first_up_size + first_down_size;

    /* Total breakdown */
    config_out->resident_bytes = total_non_expert_bytes;
    config_out->routed_bytes   = total_expert_tensor_bytes;

    /* Build moe_layer_ids array */
    config_out->moe_layer_ids = (int32_t *)malloc(
        (size_t)n_moe_layers * sizeof(int32_t)
    );
    if (config_out->moe_layer_ids) {
        int32_t idx = 0;
        for (int32_t l = 0; l <= max_layer_seen && l < MOE_CACHE_MAX_LAYERS; l++) {
            if (layer_has_routed[l] && idx < n_moe_layers) {
                config_out->moe_layer_ids[idx++] = l;
            }
        }
    }

    return AMCOLI_OK;
}

void moe_config_free(struct amcoli_moe_config *config) {
    if (!config) return;
    if (config->moe_layer_ids) {
        free(config->moe_layer_ids);
        config->moe_layer_ids = NULL;
    }
}

int32_t moe_config_validate(const struct amcoli_moe_config *config) {
    if (!config) return AMCOLI_ERR_INVALID_PARAMS;

    if (config->n_expert <= 0) {
        fprintf(stderr, "AMcoli: Invalid config: n_expert=%d\n", config->n_expert);
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    if (config->n_expert_used <= 0 || config->n_expert_used > config->n_expert) {
        fprintf(stderr, "AMcoli: Invalid config: n_expert_used=%d (n_expert=%d)\n",
            config->n_expert_used, config->n_expert);
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    if (config->n_moe_layers <= 0) {
        fprintf(stderr, "AMcoli: Invalid config: n_moe_layers=%d\n", config->n_moe_layers);
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    if (config->n_total_layers > 0 && config->n_moe_layers > config->n_total_layers) {
        fprintf(stderr, "AMcoli: Invalid config: n_moe_layers=%d > n_total_layers=%d\n",
            config->n_moe_layers, config->n_total_layers);
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    if (config->expert_total_bytes <= 0) {
        fprintf(stderr, "AMcoli: Warning: expert_total_bytes=%lld (could not determine size)\n",
            (long long)config->expert_total_bytes);
        /* Non-fatal: we might still be able to operate */
    }

    /* Validate moe_layer_ids are in range */
    if (config->moe_layer_ids && config->n_total_layers > 0) {
        for (int32_t i = 0; i < config->n_moe_layers; i++) {
            if (config->moe_layer_ids[i] < 0 ||
                config->moe_layer_ids[i] >= config->n_total_layers)
            {
                fprintf(stderr, "AMcoli: Invalid moe_layer_ids[%d]=%d (n_total_layers=%d)\n",
                    i, config->moe_layer_ids[i], config->n_total_layers);
                return AMCOLI_ERR_INVALID_PARAMS;
            }
        }
    }

    return AMCOLI_OK;
}

void moe_config_compute_budget(
    const struct amcoli_moe_config *config,
    int32_t  vram_slots,
    int32_t  ram_slots,
    int64_t *vram_needed,
    int64_t *ram_needed,
    int64_t *disk_needed
) {
    if (!config) return;

    int64_t expert_bytes = config->expert_total_bytes;
    if (expert_bytes <= 0) expert_bytes = 1; /* Avoid division by zero */

    if (vram_needed) {
        *vram_needed = (int64_t)vram_slots * expert_bytes;
    }

    if (ram_needed) {
        /* RAM needs: resident params + RAM cache slots */
        *ram_needed = config->resident_bytes + (int64_t)ram_slots * expert_bytes;
    }

    if (disk_needed) {
        /* Disk: full model file */
        *disk_needed = config->resident_bytes + config->routed_bytes;
    }
}

void moe_config_print(const struct amcoli_moe_config *config) {
    if (!config) return;

    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║     AMcoli — MoE Configuration           ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════╣\n");

    fprintf(stderr, "║  Total layers:        %6d              ║\n", config->n_total_layers);
    fprintf(stderr, "║  MoE layers:          %6d              ║\n", config->n_moe_layers);
    fprintf(stderr, "║  Routed experts/layer: %5d              ║\n", config->n_expert);
    fprintf(stderr, "║  Top-k (active):      %6d              ║\n", config->n_expert_used);
    fprintf(stderr, "║  Shared experts:      %6s              ║\n",
        config->has_shared_experts ? "yes" : "no");

    if (config->has_shared_experts) {
        fprintf(stderr, "║  Shared expert groups: %5d              ║\n",
            config->n_shared_expert);
    }

    fprintf(stderr, "╠══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Per-expert size:                        ║\n");
    fprintf(stderr, "║    Gate:  %10.2f MB                   ║\n",
        (double)config->expert_bytes_gate / (1024.0 * 1024.0));
    fprintf(stderr, "║    Up:    %10.2f MB                   ║\n",
        (double)config->expert_bytes_up / (1024.0 * 1024.0));
    fprintf(stderr, "║    Down:  %10.2f MB                   ║\n",
        (double)config->expert_bytes_down / (1024.0 * 1024.0));
    fprintf(stderr, "║    Total: %10.2f MB                   ║\n",
        (double)config->expert_total_bytes / (1024.0 * 1024.0));

    fprintf(stderr, "╠══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Model breakdown:                        ║\n");
    fprintf(stderr, "║    Resident (RAM):  %8.2f GB           ║\n",
        (double)config->resident_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "║    Routed (disk):   %8.2f GB           ║\n",
        (double)config->routed_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "║    Total:           %8.2f GB           ║\n",
        (double)(config->resident_bytes + config->routed_bytes) /
            (1024.0 * 1024.0 * 1024.0));

    double active_pct = 0.0;
    if (config->n_expert > 0) {
        active_pct = 100.0 * (double)config->n_expert_used / (double)config->n_expert;
    }
    fprintf(stderr, "║  Active ratio:      %5.1f%% of experts    ║\n", active_pct);

    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");

    /* Print MoE layer indices if verbose */
    if (config->moe_layer_ids && config->n_moe_layers <= 32) {
        fprintf(stderr, "  MoE layer indices: [");
        for (int32_t i = 0; i < config->n_moe_layers; i++) {
            fprintf(stderr, "%d", config->moe_layer_ids[i]);
            if (i < config->n_moe_layers - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n\n");
    } else if (config->moe_layer_ids) {
        fprintf(stderr, "  MoE layer indices: [%d..%d] (%d layers)\n\n",
            config->moe_layer_ids[0],
            config->moe_layer_ids[config->n_moe_layers - 1],
            config->n_moe_layers);
    }
}
