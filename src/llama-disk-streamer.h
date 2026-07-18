/**
 * @file llama-disk-streamer.h
 * @brief Disk I/O layer for on-demand expert streaming from GGUF files.
 *
 * Internal header — not part of the public API.
 *
 * Responsibilities:
 *   1. mmap the entire GGUF file at startup (read-only)
 *   2. Build an expert offset index: {layer, expert} → {offset, size}
 *   3. On cache miss: provide a pointer to expert data (mmap) or copy it
 *   4. Issue prefetch hints via madvise(MADV_WILLNEED)
 *
 * The expert offset index is built at model load time by scanning GGUF
 * tensor metadata for expert tensor name patterns:
 *   - ffn_gate_exp.{N}     (or blk.{L}.ffn_gate_exps.weight)
 *   - ffn_up_exp.{N}
 *   - ffn_down_exp.{N}
 */

#ifndef LLAMA_DISK_STREAMER_H
#define LLAMA_DISK_STREAMER_H

#include "amcoli.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Expert Record ───────────────────────────────────────────────────── */

/**
 * Location of one expert's weight tensors within the GGUF file.
 *
 * An expert consists of three tensors: gate, up, down projections.
 * They may or may not be contiguous in the file. If contiguous,
 * `is_contiguous` is true and a single read/mmap covers all three.
 */
struct expert_record {
    int32_t  layer_id;
    int32_t  expert_id;

    /* Gate projection tensor */
    int64_t  gate_offset;        /**< Byte offset from file start          */
    int64_t  gate_size;          /**< Size in bytes                        */

    /* Up projection tensor */
    int64_t  up_offset;
    int64_t  up_size;

    /* Down projection tensor */
    int64_t  down_offset;
    int64_t  down_size;

    /* Aggregate */
    int64_t  total_size;         /**< gate_size + up_size + down_size      */
    bool     is_contiguous;      /**< True if all 3 tensors are adjacent   */
    int64_t  contiguous_offset;  /**< Start offset if contiguous           */
};

/* ── Expert Offset Index ─────────────────────────────────────────────── */

/**
 * Index mapping (layer, expert) to file locations.
 *
 * This is a 2D array: records[layer_id][expert_id].
 * Built once at init by scanning GGUF tensor metadata.
 */
struct expert_offset_index {
    struct expert_record **records; /**< [n_layers][n_experts_per_layer]     */
    int32_t  n_layers;
    int32_t  n_experts;              /**< Experts per layer (uniform)        */
    int64_t  total_routed_bytes;     /**< Sum of all expert sizes            */
};

/* ── Disk Streamer ───────────────────────────────────────────────────── */

/**
 * The disk streamer manages the mmap'd GGUF file and expert offset index.
 */
struct disk_streamer {
    /* File mapping */
    const char *file_path;       /**< Path to the GGUF file                */
    int         fd;              /**< File descriptor (-1 if not open)     */
    void       *mmap_base;       /**< Base address of mmap'd region        */
    size_t      mmap_size;       /**< Total size of mmap'd region          */
    bool        mmap_active;     /**< True if mmap is live                 */

    /* Expert index */
    struct expert_offset_index index;

    /* Configuration */
    int32_t     io_backend;      /**< amcoli_io_backend enum               */
    bool        allow_mlock;     /**< Whether to mlock resident params     */

    /* Statistics */
    int64_t     stat_reads;      /**< Number of expert reads from disk     */
    int64_t     stat_bytes_read; /**< Total bytes read from disk           */
    double      stat_read_time_ms; /**< Cumulative read time in ms         */
    int64_t     stat_prefetches; /**< Number of prefetch hints issued      */
};

/* ── API ─────────────────────────────────────────────────────────────── */

/**
 * Initialize the disk streamer.
 *
 * @param streamer    Output streamer struct (caller-allocated)
 * @param params      Disk configuration
 * @return            AMCOLI_OK or error code
 *
 * Opens the GGUF file, mmaps it read-only, and builds the expert
 * offset index by scanning tensor metadata.
 */
int32_t disk_streamer_init(
    struct disk_streamer *streamer,
    const struct amcoli_disk_params *params
);

/** Free all streamer resources (closes fd, munmaps). */
void disk_streamer_free(struct disk_streamer *streamer);

/**
 * Build the expert offset index from GGUF tensor metadata.
 *
 * @param streamer     The streamer (must have mmap active)
 * @param n_layers     Number of MoE layers
 * @param n_experts    Experts per layer
 * @return             AMCOLI_OK or error code
 *
 * Scans the GGUF tensor info array for tensors matching expert naming
 * patterns and populates the offset index.
 *
 * This is called internally by disk_streamer_init but exposed for testing.
 */
int32_t disk_streamer_build_index(
    struct disk_streamer *streamer,
    int32_t n_layers,
    int32_t n_experts
);

/**
 * Read an expert's data from the mmap'd file.
 *
 * @param streamer   The streamer
 * @param layer_id   Layer index
 * @param expert_id  Expert index
 * @param dst        Destination buffer (must be >= expert total_size)
 * @param dst_size   Size of destination buffer
 * @return           Bytes copied, or -1 on error
 *
 * If the expert tensors are contiguous, this is a single memcpy.
 * If not, three separate copies are performed (gate, up, down).
 *
 * The memcpy reads from the mmap'd region, which may trigger a page
 * fault if the data isn't in the OS page cache. This is intentional —
 * the page cache acts as a transparent L3 cache layer.
 */
int64_t disk_streamer_read_expert(
    struct disk_streamer *streamer,
    int32_t  layer_id,
    int32_t  expert_id,
    void    *dst,
    size_t   dst_size
);

/**
 * Get a direct pointer to an expert's data in the mmap region.
 *
 * @param streamer   The streamer
 * @param layer_id   Layer index
 * @param expert_id  Expert index
 * @param ptr_out    Receives pointer to expert data (within mmap)
 * @param size_out   Receives total size
 * @return           AMCOLI_OK if expert is contiguous, error otherwise
 *
 * Only works for contiguous experts. For non-contiguous experts,
 * use disk_streamer_read_expert() which copies into a caller buffer.
 *
 * WARNING: The returned pointer is valid only as long as the mmap is
 * active. Do not store it across streamer lifetime boundaries.
 */
int32_t disk_streamer_get_expert_ptr(
    struct disk_streamer *streamer,
    int32_t  layer_id,
    int32_t  expert_id,
    const void **ptr_out,
    size_t      *size_out
);

/**
 * Issue a prefetch hint for the specified expert.
 *
 * @param streamer   The streamer
 * @param layer_id   Layer index
 * @param expert_id  Expert index
 *
 * mmap backend:  calls madvise(MADV_WILLNEED) on the expert's region.
 * io_uring backend: submits async read (Phase 4).
 *
 * Non-blocking. Safe to call speculatively.
 */
void disk_streamer_prefetch_expert(
    struct disk_streamer *streamer,
    int32_t layer_id,
    int32_t expert_id
);

/**
 * Get the expert record for a given (layer, expert).
 * Returns NULL if out of bounds.
 */
const struct expert_record *disk_streamer_get_record(
    const struct disk_streamer *streamer,
    int32_t layer_id,
    int32_t expert_id
);

/** Print streamer statistics to stderr. */
void disk_streamer_print_stats(const struct disk_streamer *streamer);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_DISK_STREAMER_H */
