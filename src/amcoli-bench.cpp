/**
 * @file amcoli-bench.cpp
 * @brief Benchmark harness implementation for AMcoli.
 */

#include "amcoli-bench.h"
#include "amcoli-sys-info.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

/* ── Timing helper ───────────────────────────────────────────────────── */

static double get_bench_time_ms(void) {
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

/* ── Zipfian Distribution Generator ──────────────────────────────────── */

struct zipf_sampler {
    int32_t n;
    double  exponent;
    double *cump; /* Cumulative probabilities */
};

static struct zipf_sampler *zipf_sampler_create(int32_t n, double exponent) {
    struct zipf_sampler *s = (struct zipf_sampler *)malloc(sizeof(struct zipf_sampler));
    if (!s) return NULL;

    s->n = n;
    s->exponent = exponent;
    s->cump = (double *)malloc((size_t)n * sizeof(double));
    if (!s->cump) {
        free(s);
        return NULL;
    }

    /* Compute normalization constant */
    double sum = 0.0;
    for (int32_t i = 1; i <= n; i++) {
        sum += 1.0 / pow((double)i, exponent);
    }

    /* Compute cumulative probabilities */
    double cumulative = 0.0;
    for (int32_t i = 1; i <= n; i++) {
        cumulative += (1.0 / pow((double)i, exponent)) / sum;
        s->cump[i - 1] = cumulative;
    }
    s->cump[n - 1] = 1.0; /* Ensure exactly 1.0 at end */

    return s;
}

static void zipf_sampler_free(struct zipf_sampler *s) {
    if (s) {
        free(s->cump);
        free(s);
    }
}

/**
 * Sample an index from the Zipfian distribution.
 * Returns a value in [0, n).
 */
static int32_t zipf_sample(const struct zipf_sampler *s) {
    double u = (double)rand() / (double)RAND_MAX;

    /* Binary search for cumulative probability */
    int32_t low = 0;
    int32_t high = s->n - 1;
    int32_t ans = high;

    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        if (s->cump[mid] >= u) {
            ans = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    return ans;
}

/* ── Shuffler for mapping ────────────────────────────────────────────── */

static void shuffle_array(int32_t *arr, int32_t n) {
    for (int32_t i = n - 1; i > 0; i--) {
        int32_t j = rand() % (i + 1);
        int32_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* ── Default Parameters ──────────────────────────────────────────────── */

struct amcoli_bench_params amcoli_bench_params_default(void) {
    struct amcoli_bench_params p;
    p.output_json    = NULL;
    p.num_tokens     = 100;
    p.zipf_exponent  = 1.0; /* Standard Zipfian */
    p.quiet          = false;
    return p;
}

/* ── Benchmark Runner ────────────────────────────────────────────────── */

int32_t amcoli_run_benchmark(
    struct amcoli_context *ctx,
    const struct amcoli_bench_params *params
) {
    if (!ctx || !params) return AMCOLI_ERR_INVALID_PARAMS;

    const struct amcoli_moe_config *config = amcoli_get_moe_config(ctx);
    if (!config || config->n_expert <= 0 || config->n_moe_layers <= 0) {
        return AMCOLI_ERR_NOT_MOE;
    }

    /* Seed RNG for reproducible benchmark run */
    srand(42);

    int32_t n_experts = config->n_expert;
    int32_t n_layers  = config->n_moe_layers;
    int32_t top_k     = config->n_expert_used;

    if (!params->quiet) {
        fprintf(stderr, "AMcoli Bench: Starting simulation of %d tokens...\n", params->num_tokens);
        fprintf(stderr, "              Routing: Zipfian (s=%.2f) over %d experts, top-%d\n",
            params->zipf_exponent, n_experts, top_k);
        fprintf(stderr, "              Layers: %d MoE layers\n\n", n_layers);
    }

    /* Initialize Zipf sampler */
    struct zipf_sampler *sampler = zipf_sampler_create(n_experts, params->zipf_exponent);
    if (!sampler) return AMCOLI_ERR_OUT_OF_MEMORY;

    /* Create shuffled mappings per layer so they have different hot experts */
    int32_t **layer_mappings = (int32_t **)malloc((size_t)n_layers * sizeof(int32_t *));
    if (!layer_mappings) {
        zipf_sampler_free(sampler);
        return AMCOLI_ERR_OUT_OF_MEMORY;
    }

    for (int32_t l = 0; l < n_layers; l++) {
        layer_mappings[l] = (int32_t *)malloc((size_t)n_experts * sizeof(int32_t));
        if (!layer_mappings[l]) {
            for (int32_t prev = 0; prev < l; prev++) free(layer_mappings[prev]);
            free(layer_mappings);
            zipf_sampler_free(sampler);
            return AMCOLI_ERR_OUT_OF_MEMORY;
        }

        for (int32_t e = 0; e < n_experts; e++) {
            layer_mappings[l][e] = e;
        }
        shuffle_array(layer_mappings[l], n_experts);
    }

    /* Reset stats before the benchmark run */
    amcoli_reset_stats(ctx);

    double t0 = get_bench_time_ms();

    /* Simulation loop */
    for (int32_t t = 0; t < params->num_tokens; t++) {
        for (int32_t l = 0; l < n_layers; l++) {
            int32_t layer_id = config->moe_layer_ids ?
                config->moe_layer_ids[l] : l;

            /* Pick top-k experts based on Zipfian distribution */
            int32_t selected[32]; /* Safe stack allocation for top-k */
            int32_t k_count = 0;

            while (k_count < top_k && k_count < 32) {
                int32_t zipf_idx = zipf_sample(sampler);
                int32_t expert_id = layer_mappings[l][zipf_idx];

                /* Ensure unique selections within this layer step */
                bool duplicate = false;
                for (int32_t i = 0; i < k_count; i++) {
                    if (selected[i] == expert_id) {
                        duplicate = true;
                        break;
                    }
                }

                if (!duplicate) {
                    selected[k_count++] = expert_id;
                }
            }

            /* 1. Speculative prefetch for next layer if enabled */
            if (l < n_layers - 1) {
                int32_t next_layer_id = config->moe_layer_ids ?
                    config->moe_layer_ids[l + 1] : (l + 1);

                /* For benchmark prefetch simulation, we predict the next layer's
                 * experts by sampling from its Zipfian distribution.
                 * To match ~65% prefetch accuracy, we mix 65% correct predictions
                 * with 35% random predictions. */
                int32_t predicted[32];
                for (int32_t i = 0; i < top_k; i++) {
                    if ((rand() % 100) < 65) {
                        /* Correct prediction (sampled from Zipf) */
                        int32_t zipf_idx = zipf_sample(sampler);
                        predicted[i] = layer_mappings[l + 1][zipf_idx];
                    } else {
                        /* Random/wrong prediction */
                        predicted[i] = rand() % n_experts;
                    }
                }
                amcoli_prefetch_experts(ctx, next_layer_id, predicted, top_k);
            }

            /* 2. Load the actual experts (cache check -> disk read) */
            for (int32_t i = 0; i < k_count; i++) {
                void *data = NULL;
                size_t size = 0;
                amcoli_ensure_expert(ctx, layer_id, selected[i], &data, &size);
            }
        }
    }

    double elapsed_ms = get_bench_time_ms() - t0;
    double elapsed_sec = elapsed_ms / 1000.0;

    /* Get context stats */
    struct amcoli_stats stats = amcoli_get_stats(ctx);

    /* Compute simulated throughput */
    double tok_per_sec = (double)params->num_tokens / elapsed_sec;

    if (!params->quiet) {
        fprintf(stderr, "\n=== Benchmark Complete ===\n");
        fprintf(stderr, "  Tokens simulated: %d\n", params->num_tokens);
        fprintf(stderr, "  Elapsed time:     %.2f sec\n", elapsed_sec);
        fprintf(stderr, "  Throughput:       %.2f tok/s\n", tok_per_sec);
        fprintf(stderr, "  Cache hit rate:   %.1f%% (VRAM hits: %lld, RAM hits: %lld)\n",
            stats.cache_hit_rate * 100.0,
            (long long)stats.cache_hits_vram,
            (long long)stats.cache_hits_ram);
        fprintf(stderr, "  Cache misses:     %lld (disk loads)\n", (long long)stats.cache_misses);
        fprintf(stderr, "  RAM evictions:    %lld\n", (long long)stats.evictions_ram);
        fprintf(stderr, "  VRAM evictions:   %lld\n", (long long)stats.evictions_vram);
        fprintf(stderr, "  Disk I/O read:    %.2f GB\n",
            (double)stats.disk_bytes_read / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "  Avg disk wait:    %.2f ms/miss\n", stats.disk_wait_ms_avg);
        if (stats.prefetch_hits > 0) {
            fprintf(stderr, "  Prefetch hits:    %lld\n", (long long)stats.prefetch_hits);
            fprintf(stderr, "  Prefetch wasted:  %lld\n", (long long)stats.prefetch_wasted);
        }
        fprintf(stderr, "==========================\n\n");
    }

    /* Save JSON if requested */
    if (params->output_json) {
        FILE *jf = fopen(params->output_json, "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"num_tokens\": %d,\n", params->num_tokens);
            fprintf(jf, "  \"zipf_exponent\": %.2f,\n", params->zipf_exponent);
            fprintf(jf, "  \"elapsed_sec\": %.4f,\n", elapsed_sec);
            fprintf(jf, "  \"tok_per_sec\": %.2f,\n", tok_per_sec);
            fprintf(jf, "  \"cache_hit_rate\": %.4f,\n", stats.cache_hit_rate);
            fprintf(jf, "  \"cache_hits_vram\": %lld,\n", (long long)stats.cache_hits_vram);
            fprintf(jf, "  \"cache_hits_ram\": %lld,\n", (long long)stats.cache_hits_ram);
            fprintf(jf, "  \"cache_misses\": %lld,\n", (long long)stats.cache_misses);
            fprintf(jf, "  \"evictions_ram\": %lld,\n", (long long)stats.evictions_ram);
            fprintf(jf, "  \"evictions_vram\": %lld,\n", (long long)stats.evictions_vram);
            fprintf(jf, "  \"disk_bytes_read\": %lld,\n", (long long)stats.disk_bytes_read);
            fprintf(jf, "  \"disk_wait_ms_avg\": %.3f,\n", stats.disk_wait_ms_avg);
            fprintf(jf, "  \"prefetch_hits\": %lld,\n", (long long)stats.prefetch_hits);
            fprintf(jf, "  \"prefetch_wasted\": %lld\n", (long long)stats.prefetch_wasted);
            fprintf(jf, "}\n");
            fclose(jf);
            if (!params->quiet) {
                fprintf(stderr, "AMcoli Bench: Saved JSON results to %s\n\n", params->output_json);
            }
        } else {
            fprintf(stderr, "Error: Failed to open %s for writing\n", params->output_json);
        }
    }

    /* Clean up resources */
    for (int32_t l = 0; l < n_layers; l++) free(layer_mappings[l]);
    free(layer_mappings);
    zipf_sampler_free(sampler);

    return AMCOLI_OK;
}
