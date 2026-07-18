/**
 * @file amcoli.cpp
 * @brief AMcoli context implementation — ties together cache, disk, router.
 *
 * This is the top-level glue that creates the AMcoli context by:
 *   1. Opening and mmapping the GGUF model file
 *   2. Extracting the MoE configuration from metadata
 *   3. Building the expert offset index
 *   4. Allocating the two-tier cache
 *   5. Optionally initializing the prefetch engine
 */

#include "amcoli.h"
#include "llama-moe-cache.h"
#include "llama-disk-streamer.h"
#include "llama-moe-router.h"
#include "llama-moe-prefetch.h"
#include "amcoli-sys-info.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <time.h>
#endif

/* ── Opaque context structure ────────────────────────────────────────── */

struct amcoli_context {
    struct amcoli_params       params;
    struct amcoli_moe_config   moe_config;
    struct disk_streamer       streamer;
    struct moe_cache           cache;
    struct moe_prefetch_engine prefetch;
    struct amcoli_stats        stats;

    /* Timing */
    double start_time_ms;

    bool initialized;
};

/* ── Timing ──────────────────────────────────────────────────────────── */

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

/* ── Default parameter constructors ──────────────────────────────────── */

struct amcoli_cache_params amcoli_cache_params_default(void) {
    struct amcoli_cache_params p;
    memset(&p, 0, sizeof(p));
    p.vram_budget_bytes = 0;    /* No VRAM tier by default */
    p.ram_budget_bytes  = -1;   /* Auto-detect */
    p.eviction_policy   = AMCOLI_EVICT_LRU;
    p.vram_slots        = 0;
    p.ram_slots         = 0;    /* Auto-compute from budget */
    return p;
}

struct amcoli_disk_params amcoli_disk_params_default(void) {
    struct amcoli_disk_params p;
    memset(&p, 0, sizeof(p));
    p.model_path     = NULL;
    p.io_backend     = AMCOLI_IO_MMAP;
    p.prefetch_depth = 0;
    p.allow_mlock    = false;
    return p;
}

struct amcoli_params amcoli_params_default(void) {
    struct amcoli_params p;
    memset(&p, 0, sizeof(p));
    p.cache = amcoli_cache_params_default();
    p.disk  = amcoli_disk_params_default();
    p.print_stats_on_exit = true;
    p.verbosity = 1;
    return p;
}

/* ── Context lifecycle ───────────────────────────────────────────────── */

struct amcoli_context *amcoli_context_create(
    const struct amcoli_params *params,
    int32_t *err_out
) {
    int32_t err = AMCOLI_OK;

    if (!params || !params->disk.model_path) {
        if (err_out) *err_out = AMCOLI_ERR_INVALID_PARAMS;
        return NULL;
    }

    /* Allocate context */
    struct amcoli_context *ctx =
        (struct amcoli_context *)calloc(1, sizeof(struct amcoli_context));
    if (!ctx) {
        if (err_out) *err_out = AMCOLI_ERR_OUT_OF_MEMORY;
        return NULL;
    }

    ctx->params = *params;
    ctx->start_time_ms = now_ms();

    if (params->verbosity >= 1) {
        fprintf(stderr, "AMcoli v%s — Universal MoE Disk-Streaming Engine\n",
            AMCOLI_VERSION_STRING);
    }

    /* Step 1: Initialize disk streamer (mmap the GGUF file) */
    err = disk_streamer_init(&ctx->streamer, &params->disk);
    if (err != AMCOLI_OK) {
        fprintf(stderr, "AMcoli: Failed to initialize disk streamer: %s\n",
            amcoli_error_string(err));
        free(ctx);
        if (err_out) *err_out = err;
        return NULL;
    }

    /* Step 2: Extract MoE configuration from GGUF metadata */
    err = moe_router_extract_config(
        ctx->streamer.mmap_base,
        ctx->streamer.mmap_size,
        &ctx->moe_config
    );
    if (err != AMCOLI_OK) {
        fprintf(stderr, "AMcoli: Failed to extract MoE config: %s\n",
            amcoli_error_string(err));
        disk_streamer_free(&ctx->streamer);
        free(ctx);
        if (err_out) *err_out = err;
        return NULL;
    }

    /* Validate the config */
    err = moe_config_validate(&ctx->moe_config);
    if (err != AMCOLI_OK) {
        fprintf(stderr, "AMcoli: MoE config validation failed\n");
        moe_config_free(&ctx->moe_config);
        disk_streamer_free(&ctx->streamer);
        free(ctx);
        if (err_out) *err_out = err;
        return NULL;
    }

    if (params->verbosity >= 1) {
        moe_config_print(&ctx->moe_config);
    }

    /* Step 3: Build the expert offset index */
    int32_t index_layers = ctx->moe_config.n_total_layers > 0 ?
        ctx->moe_config.n_total_layers : ctx->moe_config.n_moe_layers;

    err = disk_streamer_build_index(
        &ctx->streamer,
        index_layers,
        ctx->moe_config.n_expert
    );
    if (err != AMCOLI_OK) {
        fprintf(stderr, "AMcoli: Failed to build expert index: %s\n",
            amcoli_error_string(err));
        moe_config_free(&ctx->moe_config);
        disk_streamer_free(&ctx->streamer);
        free(ctx);
        if (err_out) *err_out = err;
        return NULL;
    }

    /* Step 4: Initialize the two-tier cache */
    struct amcoli_cache_params cache_params = params->cache;

    /* Auto-compute VRAM slots if requested */
    int64_t expert_bytes = ctx->moe_config.expert_total_bytes;
    if (expert_bytes <= 0) expert_bytes = 1;

    if (cache_params.vram_slots == 0 && cache_params.vram_budget_bytes == -1) {
        int64_t avail_vram = amcoli_sys_get_available_vram();
        int64_t safety_margin_vram = 256LL * 1024 * 1024; // 256 MB safety
        int64_t cache_budget_vram = avail_vram - safety_margin_vram;
        if (cache_budget_vram > 0) {
            int32_t vram_slots = (int32_t)(cache_budget_vram / expert_bytes);
            int32_t total_experts = ctx->moe_config.n_expert * ctx->moe_config.n_moe_layers;
            if (vram_slots > total_experts) vram_slots = total_experts;
            cache_params.vram_slots = vram_slots;
            cache_params.vram_budget_bytes = (int64_t)vram_slots * expert_bytes;
            if (params->verbosity >= 1 && vram_slots > 0) {
                fprintf(stderr, "AMcoli: Auto-sized VRAM cache: %d slots (%.2f GB) based on available VRAM (%.2f GB)\n",
                    vram_slots, (double)(vram_slots * expert_bytes) / (1024.0 * 1024.0 * 1024.0),
                    (double)avail_vram / (1024.0 * 1024.0 * 1024.0));
            }
        } else {
            cache_params.vram_slots = 0;
            cache_params.vram_budget_bytes = 0;
        }
    }

    /* Auto-compute RAM slots if requested */
    if (cache_params.ram_slots == 0 && cache_params.ram_budget_bytes == -1) {
        int64_t avail_ram = amcoli_sys_get_available_ram();
        int64_t safety_margin = 1500LL * 1024 * 1024; // 1.5 GB safety
        int64_t cache_budget = avail_ram - ctx->moe_config.resident_bytes - safety_margin;
        int32_t ram_slots = 0;
        if (cache_budget > 0) {
            ram_slots = (int32_t)(cache_budget / expert_bytes);
        }
        if (ram_slots < 4) {
            ram_slots = 4; // Absolute minimum
        }
        int32_t total_experts = ctx->moe_config.n_expert * ctx->moe_config.n_moe_layers;
        if (ram_slots > total_experts) {
            ram_slots = total_experts;
        }
        cache_params.ram_slots = ram_slots;
        cache_params.ram_budget_bytes = (int64_t)ram_slots * expert_bytes;

        if (params->verbosity >= 1) {
            fprintf(stderr, "AMcoli: Auto-sized RAM cache: %d slots (%.2f GB) based on available RAM (%.2f GB)\n",
                ram_slots, (double)(ram_slots * expert_bytes) / (1024.0 * 1024.0 * 1024.0),
                (double)avail_ram / (1024.0 * 1024.0 * 1024.0));
        }
    }

    err = moe_cache_init(
        &ctx->cache,
        &cache_params,
        (size_t)ctx->moe_config.expert_total_bytes,
        index_layers
    );
    if (err != AMCOLI_OK) {
        fprintf(stderr, "AMcoli: Failed to initialize cache: %s\n",
            amcoli_error_string(err));
        moe_config_free(&ctx->moe_config);
        disk_streamer_free(&ctx->streamer);
        free(ctx);
        if (err_out) *err_out = err;
        return NULL;
    }

    /* Step 5: Validate memory budget */
    int64_t vram_needed, ram_needed, disk_needed;
    moe_config_compute_budget(
        &ctx->moe_config,
        cache_params.vram_slots,
        cache_params.ram_slots,
        &vram_needed, &ram_needed, &disk_needed
    );

    if (params->verbosity >= 1) {
        fprintf(stderr, "AMcoli: Memory budget: VRAM=%.2f GB, RAM=%.2f GB, Disk=%.2f GB\n",
            (double)vram_needed / (1024.0 * 1024.0 * 1024.0),
            (double)ram_needed  / (1024.0 * 1024.0 * 1024.0),
            (double)disk_needed / (1024.0 * 1024.0 * 1024.0));
    }

    /* Step 6: Initialize prefetch engine */
    moe_prefetch_init(
        &ctx->prefetch,
        &ctx->streamer,
        &ctx->cache,
        params->disk.prefetch_depth
    );

    ctx->initialized = true;

    if (params->verbosity >= 1) {
        double init_time = now_ms() - ctx->start_time_ms;
        fprintf(stderr, "AMcoli: Initialization complete (%.1f ms)\n\n", init_time);
    }

    if (err_out) *err_out = AMCOLI_OK;
    return ctx;
}

void amcoli_context_free(struct amcoli_context *ctx) {
    if (!ctx) return;

    if (ctx->params.print_stats_on_exit && ctx->initialized) {
        struct amcoli_stats stats = amcoli_get_stats(ctx);
        amcoli_print_stats(&stats);
    }

    moe_prefetch_free(&ctx->prefetch);
    moe_cache_free(&ctx->cache);
    moe_config_free(&ctx->moe_config);
    disk_streamer_free(&ctx->streamer);

    free(ctx);
}

/* ── Queries ─────────────────────────────────────────────────────────── */

const struct amcoli_moe_config *amcoli_get_moe_config(
    const struct amcoli_context *ctx
) {
    if (!ctx) return NULL;
    return &ctx->moe_config;
}

struct amcoli_stats amcoli_get_stats(const struct amcoli_context *ctx) {
    struct amcoli_stats s;
    memset(&s, 0, sizeof(s));
    if (!ctx) return s;

    /* Cache stats */
    s.cache_hits_vram = ctx->cache.vram_pool.stat_hits;
    s.cache_hits_ram  = ctx->cache.ram_pool.stat_hits;
    s.cache_misses    = ctx->cache.ram_pool.stat_misses;
    s.cache_hit_rate  = moe_cache_hit_rate(&ctx->cache);
    s.evictions_vram  = ctx->cache.vram_pool.stat_evictions;
    s.evictions_ram   = ctx->cache.ram_pool.stat_evictions;

    /* Disk stats */
    s.disk_bytes_read  = ctx->streamer.stat_bytes_read;
    s.disk_wait_ms_avg = ctx->streamer.stat_reads > 0 ?
        ctx->streamer.stat_read_time_ms / (double)ctx->streamer.stat_reads : 0.0;

    /* Prefetch stats */
    s.prefetch_hits   = ctx->prefetch.stat_prefetch_hit;
    s.prefetch_wasted = ctx->prefetch.stat_prefetch_wasted;

    /* Timing */
    s.elapsed_sec = (now_ms() - ctx->start_time_ms) / 1000.0;

    /* Token stats from the internal counters */
    s.n_tokens_total = ctx->stats.n_tokens_total;

    /* Throughput */
    if (s.elapsed_sec > 0 && s.n_tokens_total > 0) {
        s.tok_per_sec_current = (double)s.n_tokens_total / s.elapsed_sec;
    }

    return s;
}

void amcoli_reset_stats(struct amcoli_context *ctx) {
    if (!ctx) return;
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    ctx->start_time_ms = now_ms();
}

void amcoli_get_cache_info(
    const struct amcoli_context *ctx,
    int32_t *vram_used,
    int32_t *vram_total,
    int32_t *ram_used,
    int32_t *ram_total
) {
    if (!ctx) return;
    if (vram_used)  *vram_used  = ctx->cache.vram_pool.n_used;
    if (vram_total) *vram_total = ctx->cache.vram_pool.n_slots;
    if (ram_used)   *ram_used   = ctx->cache.ram_pool.n_used;
    if (ram_total)  *ram_total  = ctx->cache.ram_pool.n_slots;
}

/* ── Expert Cache Operations ─────────────────────────────────────────── */

int32_t amcoli_ensure_expert(
    struct amcoli_context *ctx,
    int32_t  layer_id,
    int32_t  expert_id,
    void   **data_out,
    size_t  *size_out
) {
    if (!ctx || !ctx->initialized) return AMCOLI_ERR_INVALID_PARAMS;

    /* Try cache lookup first */
    int32_t tier;
    void *cached_data;
    size_t cached_size;

    if (moe_cache_lookup(&ctx->cache, layer_id, expert_id,
                         &tier, &cached_data, &cached_size))
    {
        /* Cache hit */
        moe_prefetch_mark_used(&ctx->prefetch, layer_id, expert_id);
        if (data_out) *data_out = cached_data;
        if (size_out) *size_out = cached_size;
        return AMCOLI_OK;
    }

    /* Cache miss — read from disk and insert into RAM cache */
    const struct expert_record *rec =
        disk_streamer_get_record(&ctx->streamer, layer_id, expert_id);
    if (!rec || rec->total_size == 0) {
        return AMCOLI_ERR_IO_FAILED;
    }

    /* Allocate temporary buffer for disk read */
    void *temp_buf = malloc((size_t)rec->total_size);
    if (!temp_buf) return AMCOLI_ERR_OUT_OF_MEMORY;

    /* Read from disk */
    int64_t bytes_read = disk_streamer_read_expert(
        &ctx->streamer, layer_id, expert_id,
        temp_buf, (size_t)rec->total_size
    );
    if (bytes_read <= 0) {
        free(temp_buf);
        return AMCOLI_ERR_IO_FAILED;
    }

    /* Insert into RAM cache (TODO: VRAM cache for GPU path) */
    void *cache_ptr = NULL;
    int32_t err = moe_cache_insert(
        &ctx->cache, MOE_TIER_RAM,
        layer_id, expert_id,
        temp_buf, (size_t)bytes_read,
        &cache_ptr
    );

    free(temp_buf);

    if (err != AMCOLI_OK) {
        return err;
    }

    if (data_out) *data_out = cache_ptr;
    if (size_out) *size_out = (size_t)bytes_read;
    return AMCOLI_OK;
}

void amcoli_prefetch_experts(
    struct amcoli_context *ctx,
    int32_t        layer_id,
    const int32_t *expert_ids,
    int32_t        n_experts
) {
    if (!ctx || !ctx->initialized) return;
    moe_prefetch_issue(&ctx->prefetch, layer_id, expert_ids, n_experts);
}

/* ── Utility ─────────────────────────────────────────────────────────── */

void amcoli_print_moe_config(const struct amcoli_moe_config *config) {
    moe_config_print(config);
}

void amcoli_print_stats(const struct amcoli_stats *stats) {
    if (!stats) return;

    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║     AMcoli — Inference Statistics         ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════╣\n");

    fprintf(stderr, "║  Tokens processed: %8lld              ║\n",
        (long long)stats->n_tokens_total);
    fprintf(stderr, "║  Elapsed:          %8.1f sec           ║\n",
        stats->elapsed_sec);

    if (stats->tok_per_sec_current > 0) {
        fprintf(stderr, "║  Throughput:       %8.2f tok/s        ║\n",
            stats->tok_per_sec_current);
    }

    fprintf(stderr, "╠══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Cache Performance:                      ║\n");
    fprintf(stderr, "║    VRAM hits:    %10lld               ║\n",
        (long long)stats->cache_hits_vram);
    fprintf(stderr, "║    RAM hits:     %10lld               ║\n",
        (long long)stats->cache_hits_ram);
    fprintf(stderr, "║    Misses:       %10lld               ║\n",
        (long long)stats->cache_misses);
    fprintf(stderr, "║    Hit rate:     %9.1f%%               ║\n",
        stats->cache_hit_rate * 100.0);
    fprintf(stderr, "║    VRAM evicts:  %10lld               ║\n",
        (long long)stats->evictions_vram);
    fprintf(stderr, "║    RAM evicts:   %10lld               ║\n",
        (long long)stats->evictions_ram);

    fprintf(stderr, "╠══════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Disk I/O:                               ║\n");
    fprintf(stderr, "║    Bytes read:   %8.2f GB              ║\n",
        (double)stats->disk_bytes_read / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "║    Avg wait:     %8.2f ms              ║\n",
        stats->disk_wait_ms_avg);

    if (stats->prefetch_hits + stats->prefetch_wasted > 0) {
        fprintf(stderr, "╠══════════════════════════════════════════╣\n");
        fprintf(stderr, "║  Prefetch:                               ║\n");
        fprintf(stderr, "║    Hits:         %10lld               ║\n",
            (long long)stats->prefetch_hits);
        fprintf(stderr, "║    Wasted:       %10lld               ║\n",
            (long long)stats->prefetch_wasted);
    }

    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
}

const char *amcoli_error_string(int32_t err) {
    switch (err) {
        case AMCOLI_OK:                  return "OK";
        case AMCOLI_ERR_INVALID_PARAMS:  return "Invalid parameters";
        case AMCOLI_ERR_NOT_MOE:         return "Model is not a Mixture-of-Experts model";
        case AMCOLI_ERR_BUDGET_EXCEEDED: return "Model exceeds declared memory budget";
        case AMCOLI_ERR_MMAP_FAILED:     return "Failed to memory-map model file";
        case AMCOLI_ERR_IO_FAILED:       return "I/O error reading model file";
        case AMCOLI_ERR_OUT_OF_MEMORY:   return "Out of memory";
        case AMCOLI_ERR_UNSUPPORTED:     return "Feature not supported/compiled";
        default:                         return "Unknown error";
    }
}
