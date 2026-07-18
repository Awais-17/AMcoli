/**
 * @file llama-disk-streamer.cpp
 * @brief Disk I/O implementation for on-demand expert streaming.
 *
 * Implements mmap-based disk streaming with an expert offset index.
 * The GGUF file is memory-mapped at startup. Expert data is accessed
 * through the mmap region — page faults handle the actual disk I/O
 * transparently, with the OS page cache acting as an additional cache tier.
 *
 * Platform notes:
 *   - Linux:   uses mmap, madvise(MADV_WILLNEED), mlock
 *   - Windows: uses CreateFileMapping/MapViewOfFile, PrefetchVirtualMemory
 *              (behind #ifdef _WIN32)
 */

#include "llama-disk-streamer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ── Platform-specific includes ──────────────────────────────────────── */

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <io.h>
    #include <fcntl.h>
    #include <sys/stat.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <time.h>
#endif

/* ── Timing helper ───────────────────────────────────────────────────── */

static double get_time_ms(void) {
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

/* ── GGUF Minimal Parsing ────────────────────────────────────────────── */

/*
 * We need just enough GGUF parsing to locate tensor metadata.
 *
 * GGUF file layout:
 *   [Header]
 *     magic: "GGUF" (4 bytes)
 *     version: uint32
 *     n_tensors: uint64
 *     n_kv: uint64
 *   [KV pairs] × n_kv
 *   [Tensor infos] × n_tensors
 *   [Tensor data] (aligned)
 *
 * Each tensor info contains:
 *   name (string), n_dims, dims[], type (enum), offset (relative to data start)
 */

#define GGUF_MAGIC 0x46475547  /* "GGUF" in little-endian */

/* GGUF value types */
enum gguf_type {
    GGUF_TYPE_UINT8    = 0,
    GGUF_TYPE_INT8     = 1,
    GGUF_TYPE_UINT16   = 2,
    GGUF_TYPE_INT16    = 3,
    GGUF_TYPE_UINT32   = 4,
    GGUF_TYPE_INT32    = 5,
    GGUF_TYPE_FLOAT32  = 6,
    GGUF_TYPE_BOOL     = 7,
    GGUF_TYPE_STRING   = 8,
    GGUF_TYPE_ARRAY    = 9,
    GGUF_TYPE_UINT64   = 10,
    GGUF_TYPE_INT64    = 11,
    GGUF_TYPE_FLOAT64  = 12,
};

/* GGUF tensor types (quantization formats) — sizes in bits per weight */
static const int gguf_type_size_bits[] = {
    /*  0 F32   */  32,
    /*  1 F16   */  16,
    /*  2 Q4_0  */   4,  /* approximate — actual is 4.5 with group overhead */
    /*  3 Q4_1  */   5,
    /*  4 Q5_0  */   5,
    /*  5 Q5_1  */   6,
    /*  6 Q8_0  */   8,
    /*  7 Q8_1  */   9,
    /*  8 Q2_K  */   3,
    /*  9 Q3_K  */   3,
    /* 10 Q4_K  */   5,
    /* 11 Q5_K  */   6,
    /* 12 Q6_K  */   6,
    /* 13 Q8_K  */   8,
    /* 14 IQ2   */   2,
    /* 15 IQ3   */   3,
    /* 16 IQ1   */   2,
    /* 17 IQ4_NL*/   5,
    /* 18 IQ4_XS*/   4,
    /* 19 IQ1_M */   2,
    /* 20 BF16  */  16,
};
#define GGUF_N_TENSOR_TYPES 21

/**
 * Minimal reader cursor for walking the mmap'd GGUF data.
 */
struct gguf_reader {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
};

static bool reader_has(const struct gguf_reader *r, size_t n) {
    return (r->pos + n) <= r->size;
}

static uint8_t read_u8(struct gguf_reader *r) {
    if (!reader_has(r, 1)) return 0;
    uint8_t v = r->base[r->pos];
    r->pos += 1;
    return v;
}

static uint32_t read_u32(struct gguf_reader *r) {
    if (!reader_has(r, 4)) return 0;
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(struct gguf_reader *r) {
    if (!reader_has(r, 8)) return 0;
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static int32_t read_i32(struct gguf_reader *r) {
    return (int32_t)read_u32(r);
}

static float read_f32(struct gguf_reader *r) {
    if (!reader_has(r, 4)) return 0.0f;
    float v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

/**
 * Read a GGUF string: uint64 length + char[length].
 * Returns a pointer into the mmap (no allocation). Length written to len_out.
 */
static const char *read_string(struct gguf_reader *r, uint64_t *len_out) {
    uint64_t len = read_u64(r);
    if (len_out) *len_out = len;
    if (!reader_has(r, (size_t)len)) {
        if (len_out) *len_out = 0;
        return NULL;
    }
    const char *s = (const char *)(r->base + r->pos);
    r->pos += (size_t)len;
    return s;
}

/**
 * Skip a GGUF KV value of the given type.
 * Returns false on error (unexpected type / out of bounds).
 */
static bool skip_gguf_value(struct gguf_reader *r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            r->pos += 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            r->pos += 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            r->pos += 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            uint64_t len;
            read_string(r, &len);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            for (uint64_t i = 0; i < arr_len; i++) {
                if (!skip_gguf_value(r, arr_type)) return false;
            }
            break;
        }
        default:
            return false;
    }
    return reader_has(r, 0); /* Just check we haven't overrun */
}

/* ── Expert tensor name parsing ──────────────────────────────────────── */

/**
 * Parse expert tensor names to extract layer and expert IDs.
 *
 * Common patterns across architectures:
 *   "blk.{L}.ffn_gate_exps.weight"          → (layer=L, type=gate)
 *   "blk.{L}.ffn_up_exps.weight"            → (layer=L, type=up)
 *   "blk.{L}.ffn_down_exps.weight"          → (layer=L, type=down)
 *   "model.layers.{L}.mlp.experts.{E}.gate_proj.weight"  → individual expert
 *
 * For merged expert tensors (exps plural), the tensor contains all experts
 * for that layer concatenated. Individual expert data is at:
 *   offset + expert_id * (tensor_size / n_experts)
 *
 * Returns true if the name matches an expert pattern.
 */
enum expert_tensor_type {
    EXPERT_TENSOR_NONE = 0,
    EXPERT_TENSOR_GATE = 1,
    EXPERT_TENSOR_UP   = 2,
    EXPERT_TENSOR_DOWN = 3,
};

struct parsed_expert_tensor {
    int32_t layer_id;
    int32_t expert_id;         /* -1 for merged (plural) tensors           */
    enum expert_tensor_type type;
    bool    is_merged;         /* True if tensor is "exps" (all experts)   */
};

/**
 * Try to parse a tensor name as an expert tensor.
 * Handles both "blk.{L}.ffn_{gate,up,down}_exps" and per-expert patterns.
 */
static bool parse_expert_tensor_name(
    const char *name,
    size_t      name_len,
    struct parsed_expert_tensor *out
) {
    if (!name || name_len == 0 || !out) return false;

    memset(out, 0, sizeof(*out));
    out->expert_id = -1;

    /* Pattern: "blk.{L}.ffn_{gate,up,down}_exp{s}" */
    if (name_len > 4 && strncmp(name, "blk.", 4) == 0) {
        /* Extract layer number */
        const char *p = name + 4;
        char *end;
        long layer = strtol(p, &end, 10);
        if (end == p || *end != '.') return false;
        out->layer_id = (int32_t)layer;

        p = end + 1; /* Skip '.' */
        size_t remaining = name_len - (size_t)(p - name);

        /* Check for ffn_gate_exp, ffn_up_exp, ffn_down_exp */
        if (remaining >= 12 && strncmp(p, "ffn_gate_exp", 12) == 0) {
            out->type = EXPERT_TENSOR_GATE;
        } else if (remaining >= 10 && strncmp(p, "ffn_up_exp", 10) == 0) {
            out->type = EXPERT_TENSOR_UP;
        } else if (remaining >= 12 && strncmp(p, "ffn_down_exp", 12) == 0) {
            out->type = EXPERT_TENSOR_DOWN;
        } else {
            return false;
        }

        /* Check if merged (plural: "exps") or individual ("exp.{N}") */
        const char *after_exp = p;
        while (after_exp < name + name_len && *after_exp != 's' && *after_exp != '.') {
            after_exp++;
        }

        /* Look for "exps" (merged) vs "exp.N" (individual) */
        /* Simple heuristic: if name contains "exps", it's merged */
        if (strstr(p, "exps") != NULL) {
            out->is_merged = true;
            out->expert_id = -1;
        } else {
            /* Try to find ".{N}" after "exp" */
            const char *dot = strstr(p, "exp.");
            if (dot) {
                dot += 4; /* skip "exp." */
                long eid = strtol(dot, &end, 10);
                if (end != dot) {
                    out->is_merged = false;
                    out->expert_id = (int32_t)eid;
                }
            }
        }

        return true;
    }

    return false;
}

/* ── Platform mmap wrappers ──────────────────────────────────────────── */

#ifdef _WIN32

static HANDLE g_file_mapping = NULL;

static void *platform_mmap(const char *path, size_t *size_out, int *fd_out) {
    HANDLE hFile = CreateFileA(
        path, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        return NULL;
    }
    *size_out = (size_t)file_size.QuadPart;

    g_file_mapping = CreateFileMappingA(
        hFile, NULL, PAGE_READONLY, 0, 0, NULL
    );
    if (!g_file_mapping) {
        CloseHandle(hFile);
        return NULL;
    }

    void *base = MapViewOfFile(g_file_mapping, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(g_file_mapping);
        CloseHandle(hFile);
        g_file_mapping = NULL;
        return NULL;
    }

    /* Store the file handle as fd (cast) for later cleanup */
    *fd_out = (int)(intptr_t)hFile;
    return base;
}

static void platform_munmap(void *base, size_t size, int fd) {
    (void)size;
    if (base) UnmapViewOfFile(base);
    if (g_file_mapping) { CloseHandle(g_file_mapping); g_file_mapping = NULL; }
    if (fd != -1) CloseHandle((HANDLE)(intptr_t)fd);
}

static void platform_prefetch(void *base, int64_t offset, int64_t size) {
    /* On Windows, PrefetchVirtualMemory is available on Win8+.
     * For simplicity, we just touch the first byte of each page. */
    (void)base; (void)offset; (void)size;
    /* TODO: Implement PrefetchVirtualMemory for Windows */
}

#else /* POSIX */

static void *platform_mmap(const char *path, size_t *size_out, int *fd_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    *size_out = (size_t)st.st_size;

    void *base = mmap(NULL, *size_out, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    /* Advise sequential access for initial scan, then random for inference */
    madvise(base, *size_out, MADV_RANDOM);

    *fd_out = fd;
    return base;
}

static void platform_munmap(void *base, size_t size, int fd) {
    if (base) munmap(base, size);
    if (fd >= 0) close(fd);
}

static void platform_prefetch(void *base, int64_t offset, int64_t size) {
    if (!base || size <= 0) return;
    madvise((char *)base + offset, (size_t)size, MADV_WILLNEED);
}

#endif /* _WIN32 */

/* ── API Implementation ──────────────────────────────────────────────── */

int32_t disk_streamer_init(
    struct disk_streamer *streamer,
    const struct amcoli_disk_params *params
) {
    if (!streamer || !params || !params->model_path) {
        return AMCOLI_ERR_INVALID_PARAMS;
    }

    memset(streamer, 0, sizeof(*streamer));
    streamer->fd = -1;
    streamer->io_backend  = params->io_backend;
    streamer->allow_mlock  = params->allow_mlock;

    /* Duplicate the path string */
    size_t path_len = strlen(params->model_path);
    char *path_copy = (char *)malloc(path_len + 1);
    if (!path_copy) return AMCOLI_ERR_OUT_OF_MEMORY;
    memcpy(path_copy, params->model_path, path_len + 1);
    streamer->file_path = path_copy;

    /* mmap the file */
    size_t mmap_size = 0;
    int fd = -1;
    void *base = platform_mmap(params->model_path, &mmap_size, &fd);
    if (!base) {
        free(path_copy);
        streamer->file_path = NULL;
        return AMCOLI_ERR_MMAP_FAILED;
    }

    streamer->mmap_base  = base;
    streamer->mmap_size  = mmap_size;
    streamer->fd         = fd;
    streamer->mmap_active = true;

    /* Validate GGUF magic */
    if (mmap_size < 24) {
        disk_streamer_free(streamer);
        return AMCOLI_ERR_IO_FAILED;
    }
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "AMcoli: Not a GGUF file (magic=0x%08x, expected 0x%08x)\n",
            magic, GGUF_MAGIC);
        disk_streamer_free(streamer);
        return AMCOLI_ERR_IO_FAILED;
    }

    fprintf(stderr, "AMcoli: Mapped GGUF file: %s (%.2f GB)\n",
        params->model_path, (double)mmap_size / (1024.0 * 1024.0 * 1024.0));

    return AMCOLI_OK;
}

void disk_streamer_free(struct disk_streamer *streamer) {
    if (!streamer) return;

    if (streamer->mmap_active) {
        platform_munmap(streamer->mmap_base, streamer->mmap_size, streamer->fd);
        streamer->mmap_base   = NULL;
        streamer->mmap_active = false;
        streamer->fd          = -1;
    }

    /* Free the expert offset index */
    if (streamer->index.records) {
        for (int32_t l = 0; l < streamer->index.n_layers; l++) {
            free(streamer->index.records[l]);
        }
        free(streamer->index.records);
        streamer->index.records = NULL;
    }

    if (streamer->file_path) {
        free((void *)streamer->file_path);
        streamer->file_path = NULL;
    }

    memset(streamer, 0, sizeof(*streamer));
    streamer->fd = -1;
}

int32_t disk_streamer_build_index(
    struct disk_streamer *streamer,
    int32_t n_layers,
    int32_t n_experts
) {
    if (!streamer || !streamer->mmap_active) return AMCOLI_ERR_INVALID_PARAMS;
    if (n_layers <= 0 || n_experts <= 0) return AMCOLI_ERR_INVALID_PARAMS;

    /* Allocate the 2D index */
    streamer->index.n_layers  = n_layers;
    streamer->index.n_experts = n_experts;
    streamer->index.records = (struct expert_record **)calloc(
        (size_t)n_layers, sizeof(struct expert_record *)
    );
    if (!streamer->index.records) return AMCOLI_ERR_OUT_OF_MEMORY;

    for (int32_t l = 0; l < n_layers; l++) {
        streamer->index.records[l] = (struct expert_record *)calloc(
            (size_t)n_experts, sizeof(struct expert_record)
        );
        if (!streamer->index.records[l]) return AMCOLI_ERR_OUT_OF_MEMORY;

        for (int32_t e = 0; e < n_experts; e++) {
            streamer->index.records[l][e].layer_id  = l;
            streamer->index.records[l][e].expert_id = e;
        }
    }

    /* Parse GGUF header to find tensor infos */
    struct gguf_reader r;
    r.base = (const uint8_t *)streamer->mmap_base;
    r.size = streamer->mmap_size;
    r.pos  = 0;

    /* Skip magic */
    uint32_t magic = read_u32(&r);
    (void)magic;

    uint32_t version   = read_u32(&r);
    uint64_t n_tensors = read_u64(&r);
    uint64_t n_kv      = read_u64(&r);

    (void)version;

    /* Skip KV pairs to reach tensor infos */
    for (uint64_t i = 0; i < n_kv; i++) {
        /* key: string */
        uint64_t key_len;
        read_string(&r, &key_len);

        /* value type */
        uint32_t vtype = read_u32(&r);

        /* skip value */
        if (!skip_gguf_value(&r, vtype)) {
            fprintf(stderr, "AMcoli: Error parsing GGUF KV at index %llu\n",
                (unsigned long long)i);
            return AMCOLI_ERR_IO_FAILED;
        }
    }

    /* Now at tensor info section. Parse each tensor info. */
    /*
     * Tensor info format:
     *   name: string (uint64 len + chars)
     *   n_dims: uint32
     *   dims: uint64[n_dims]
     *   type: uint32
     *   offset: uint64 (relative to tensor data start)
     */

    size_t tensor_infos_end = r.pos; /* Will be computed after reading all infos */

    struct {
        char     name[256];
        size_t   name_len;
        uint32_t n_dims;
        uint64_t dims[4];
        uint32_t type;
        uint64_t offset;
        uint64_t computed_size; /* Size in bytes */
    } *tensor_infos = NULL;

    tensor_infos = (decltype(tensor_infos))calloc((size_t)n_tensors, sizeof(*tensor_infos));
    if (!tensor_infos) return AMCOLI_ERR_OUT_OF_MEMORY;

    for (uint64_t t = 0; t < n_tensors; t++) {
        /* Read tensor name */
        uint64_t name_len;
        const char *name = read_string(&r, &name_len);
        if (name_len >= 256) name_len = 255;
        if (name) {
            memcpy(tensor_infos[t].name, name, (size_t)name_len);
        }
        tensor_infos[t].name[name_len] = '\0';
        tensor_infos[t].name_len = (size_t)name_len;

        /* n_dims */
        tensor_infos[t].n_dims = read_u32(&r);

        /* dims */
        for (uint32_t d = 0; d < tensor_infos[t].n_dims && d < 4; d++) {
            tensor_infos[t].dims[d] = read_u64(&r);
        }

        /* type */
        tensor_infos[t].type = read_u32(&r);

        /* offset (relative to data section start) */
        tensor_infos[t].offset = read_u64(&r);

        /* Compute size from dims and type */
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < tensor_infos[t].n_dims; d++) {
            n_elements *= tensor_infos[t].dims[d];
        }

        int bits_per_weight = 32; /* Default to F32 */
        if (tensor_infos[t].type < GGUF_N_TENSOR_TYPES) {
            bits_per_weight = gguf_type_size_bits[tensor_infos[t].type];
        }
        tensor_infos[t].computed_size = (n_elements * (uint64_t)bits_per_weight + 7) / 8;
    }

    /* The data section starts after tensor infos, aligned to 32 bytes */
    tensor_infos_end = r.pos;
    size_t data_start = (tensor_infos_end + 31) & ~(size_t)31;

    /* Now populate the expert offset index */
    int64_t total_routed = 0;
    int32_t experts_found = 0;

    for (uint64_t t = 0; t < n_tensors; t++) {
        struct parsed_expert_tensor parsed;
        if (!parse_expert_tensor_name(
            tensor_infos[t].name,
            tensor_infos[t].name_len,
            &parsed))
        {
            continue; /* Not an expert tensor */
        }

        if (parsed.layer_id < 0 || parsed.layer_id >= n_layers) continue;

        int64_t abs_offset = (int64_t)(data_start + tensor_infos[t].offset);
        int64_t tensor_size = (int64_t)tensor_infos[t].computed_size;

        if (parsed.is_merged) {
            /* Merged tensor: contains all experts concatenated.
             * Each expert's portion is tensor_size / n_experts. */
            int64_t per_expert_size = tensor_size / n_experts;

            for (int32_t e = 0; e < n_experts; e++) {
                struct expert_record *rec =
                    &streamer->index.records[parsed.layer_id][e];

                int64_t expert_offset = abs_offset + (int64_t)e * per_expert_size;

                switch (parsed.type) {
                    case EXPERT_TENSOR_GATE:
                        rec->gate_offset = expert_offset;
                        rec->gate_size   = per_expert_size;
                        break;
                    case EXPERT_TENSOR_UP:
                        rec->up_offset = expert_offset;
                        rec->up_size   = per_expert_size;
                        break;
                    case EXPERT_TENSOR_DOWN:
                        rec->down_offset = expert_offset;
                        rec->down_size   = per_expert_size;
                        break;
                    default: break;
                }
            }
            experts_found++;
        } else if (parsed.expert_id >= 0 && parsed.expert_id < n_experts) {
            /* Individual expert tensor */
            struct expert_record *rec =
                &streamer->index.records[parsed.layer_id][parsed.expert_id];

            switch (parsed.type) {
                case EXPERT_TENSOR_GATE:
                    rec->gate_offset = abs_offset;
                    rec->gate_size   = tensor_size;
                    break;
                case EXPERT_TENSOR_UP:
                    rec->up_offset = abs_offset;
                    rec->up_size   = tensor_size;
                    break;
                case EXPERT_TENSOR_DOWN:
                    rec->down_offset = abs_offset;
                    rec->down_size   = tensor_size;
                    break;
                default: break;
            }
            experts_found++;
        }
    }

    /* Finalize records: compute totals and contiguity */
    for (int32_t l = 0; l < n_layers; l++) {
        for (int32_t e = 0; e < n_experts; e++) {
            struct expert_record *rec = &streamer->index.records[l][e];
            rec->total_size = rec->gate_size + rec->up_size + rec->down_size;
            total_routed += rec->total_size;

            /* Check contiguity: gate + up + down must be adjacent */
            if (rec->gate_size > 0 && rec->up_size > 0 && rec->down_size > 0) {
                if (rec->gate_offset + rec->gate_size == rec->up_offset &&
                    rec->up_offset + rec->up_size == rec->down_offset)
                {
                    rec->is_contiguous = true;
                    rec->contiguous_offset = rec->gate_offset;
                } else {
                    rec->is_contiguous = false;
                }
            }
        }
    }

    streamer->index.total_routed_bytes = total_routed;

    fprintf(stderr, "AMcoli: Built expert index: %d layers × %d experts, "
        "%.2f GB routed, %d tensor groups found\n",
        n_layers, n_experts,
        (double)total_routed / (1024.0 * 1024.0 * 1024.0),
        experts_found);

    free(tensor_infos);
    return AMCOLI_OK;
}

int64_t disk_streamer_read_expert(
    struct disk_streamer *streamer,
    int32_t  layer_id,
    int32_t  expert_id,
    void    *dst,
    size_t   dst_size
) {
    if (!streamer || !streamer->mmap_active || !dst) return -1;

    const struct expert_record *rec =
        disk_streamer_get_record(streamer, layer_id, expert_id);
    if (!rec || rec->total_size == 0) return -1;
    if ((int64_t)dst_size < rec->total_size) return -1;

    double t0 = get_time_ms();

    if (rec->is_contiguous) {
        /* Single memcpy for all three tensors */
        memcpy(dst,
               (const char *)streamer->mmap_base + rec->contiguous_offset,
               (size_t)rec->total_size);
    } else {
        /* Three separate copies */
        char *p = (char *)dst;

        if (rec->gate_size > 0) {
            memcpy(p, (const char *)streamer->mmap_base + rec->gate_offset,
                   (size_t)rec->gate_size);
            p += rec->gate_size;
        }
        if (rec->up_size > 0) {
            memcpy(p, (const char *)streamer->mmap_base + rec->up_offset,
                   (size_t)rec->up_size);
            p += rec->up_size;
        }
        if (rec->down_size > 0) {
            memcpy(p, (const char *)streamer->mmap_base + rec->down_offset,
                   (size_t)rec->down_size);
        }
    }

    double elapsed = get_time_ms() - t0;

    streamer->stat_reads++;
    streamer->stat_bytes_read += rec->total_size;
    streamer->stat_read_time_ms += elapsed;

    return rec->total_size;
}

int32_t disk_streamer_get_expert_ptr(
    struct disk_streamer *streamer,
    int32_t  layer_id,
    int32_t  expert_id,
    const void **ptr_out,
    size_t      *size_out
) {
    if (!streamer || !streamer->mmap_active) return AMCOLI_ERR_INVALID_PARAMS;

    const struct expert_record *rec =
        disk_streamer_get_record(streamer, layer_id, expert_id);
    if (!rec) return AMCOLI_ERR_INVALID_PARAMS;

    if (!rec->is_contiguous) {
        return AMCOLI_ERR_UNSUPPORTED; /* Must use read_expert for non-contiguous */
    }

    if (ptr_out) {
        *ptr_out = (const char *)streamer->mmap_base + rec->contiguous_offset;
    }
    if (size_out) {
        *size_out = (size_t)rec->total_size;
    }

    return AMCOLI_OK;
}

void disk_streamer_prefetch_expert(
    struct disk_streamer *streamer,
    int32_t layer_id,
    int32_t expert_id
) {
    if (!streamer || !streamer->mmap_active) return;

    const struct expert_record *rec =
        disk_streamer_get_record(streamer, layer_id, expert_id);
    if (!rec || rec->total_size == 0) return;

    if (rec->is_contiguous) {
        platform_prefetch(streamer->mmap_base,
                          rec->contiguous_offset, rec->total_size);
    } else {
        /* Prefetch each tensor region separately */
        if (rec->gate_size > 0)
            platform_prefetch(streamer->mmap_base, rec->gate_offset, rec->gate_size);
        if (rec->up_size > 0)
            platform_prefetch(streamer->mmap_base, rec->up_offset, rec->up_size);
        if (rec->down_size > 0)
            platform_prefetch(streamer->mmap_base, rec->down_offset, rec->down_size);
    }

    streamer->stat_prefetches++;
}

const struct expert_record *disk_streamer_get_record(
    const struct disk_streamer *streamer,
    int32_t layer_id,
    int32_t expert_id
) {
    if (!streamer || !streamer->index.records) return NULL;
    if (layer_id < 0 || layer_id >= streamer->index.n_layers) return NULL;
    if (expert_id < 0 || expert_id >= streamer->index.n_experts) return NULL;
    return &streamer->index.records[layer_id][expert_id];
}

void disk_streamer_print_stats(const struct disk_streamer *streamer) {
    if (!streamer) return;

    fprintf(stderr, "\n=== AMcoli Disk Streamer Statistics ===\n");
    fprintf(stderr, "  File: %s\n", streamer->file_path ? streamer->file_path : "(none)");
    fprintf(stderr, "  Mapped size: %.2f GB\n",
        (double)streamer->mmap_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Expert reads: %lld\n", (long long)streamer->stat_reads);
    fprintf(stderr, "  Bytes read: %.2f GB\n",
        (double)streamer->stat_bytes_read / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Total read time: %.1f ms\n", streamer->stat_read_time_ms);
    if (streamer->stat_reads > 0) {
        fprintf(stderr, "  Avg read time: %.2f ms/expert\n",
            streamer->stat_read_time_ms / (double)streamer->stat_reads);
    }
    fprintf(stderr, "  Prefetch hints: %lld\n", (long long)streamer->stat_prefetches);

    if (streamer->index.records) {
        fprintf(stderr, "  Index: %d layers × %d experts = %.2f GB routed\n",
            streamer->index.n_layers,
            streamer->index.n_experts,
            (double)streamer->index.total_routed_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    fprintf(stderr, "======================================\n\n");
}
