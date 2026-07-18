/**
 * @file amcoli-bench.h
 * @brief Benchmark harness for AMcoli.
 *
 * Simulates real MoE inference patterns (using Zipfian distribution to
 * model expert locality) and measures cache hits, misses, disk I/O,
 * and throughput under configurable budgets.
 */

#ifndef AMCOLI_BENCH_H
#define AMCOLI_BENCH_H

#include "amcoli.h"

#ifdef __cplusplus
extern "C" {
#endif

struct amcoli_bench_params {
    const char *output_json;    /**< Path to save JSON results, or NULL */
    int32_t     num_tokens;     /**< Number of tokens to simulate       */
    double      zipf_exponent;  /**< Exponent for Zipfian routing (locality) */
    bool        quiet;          /**< If true, suppress stdout logs      */
};

/** Returns sensible default benchmark parameters. */
struct amcoli_bench_params amcoli_bench_params_default(void);

/**
 * Run the benchmark harness on an initialized AMcoli context.
 *
 * @param ctx     Initialized AMcoli context
 * @param params  Benchmark parameters
 * @return        AMCOLI_OK or error code
 */
int32_t amcoli_run_benchmark(
    struct amcoli_context *ctx,
    const struct amcoli_bench_params *params
);

#ifdef __cplusplus
}
#endif

#endif /* AMCOLI_BENCH_H */
