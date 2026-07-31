/**
 * @file amcoli-sys-info.cpp
 * @brief Platform-specific system information implementation.
 */

#include "amcoli-sys-info.h"
#include "amcoli-downloader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    /* For DXGI dynamic loading */
    typedef HRESULT (WINAPI *LPCREATEDXGIFACTORY)(REFIID, void**);
#else
    #include <sys/sysinfo.h>
    #include <unistd.h>
#endif

/* ── RAM Detection ───────────────────────────────────────────────────── */

int64_t amcoli_sys_get_total_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return (int64_t)status.ullTotalPhys;
    }
    return 0;
#else
    /* Try /proc/meminfo first for accuracy */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        int64_t val = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                if (sscanf(line + 9, "%lld", &val) == 1) {
                    val *= 1024; /* kB to bytes */
                }
                break;
            }
        }
        fclose(f);
        if (val > 0) return val;
    }

    /* Fallback to sysinfo */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (int64_t)si.totalram * si.mem_unit;
    }
    return 0;
#endif
}

int64_t amcoli_sys_get_available_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return (int64_t)status.ullAvailPhys;
    }
    return 0;
#else
    /* Try /proc/meminfo first for MemAvailable */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        int64_t val = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                if (sscanf(line + 13, "%lld", &val) == 1) {
                    val *= 1024;
                }
                break;
            }
        }
        fclose(f);
        if (val > 0) return val;
    }

    /* Fallback to sysinfo freeram + bufferram */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (int64_t)(si.freeram + si.bufferram) * si.mem_unit;
    }
    return 0;
#endif
}

/* ── VRAM Detection ──────────────────────────────────────────────────── */

#ifdef _WIN32
/* Helper to query DXGI memory dynamically to avoid link dependencies */
static int64_t query_dxgi_vram(bool available_only) {
    HMODULE hDxgi = LoadLibraryA("dxgi.dll");
    if (!hDxgi) return 0;

    LPCREATEDXGIFACTORY pCreateDXGIFactory =
        (LPCREATEDXGIFACTORY)(void*)GetProcAddress(hDxgi, "CreateDXGIFactory");
    if (!pCreateDXGIFactory) {
        FreeLibrary(hDxgi);
        return 0;
    }

    /* We need to define DXGI interfaces locally to avoid dxgi.h dependency */
    struct MyDXGI_ADAPTER_DESC {
        WCHAR Description[128];
        UINT VendorId;
        UINT DeviceId;
        UINT SubSysId;
        UINT Revision;
        SIZE_T DedicatedVideoMemory;
        SIZE_T DedicatedSystemMemory;
        SIZE_T SharedSystemMemory;
        LUID AdapterLuid;
    };

    /* UUIDs for DXGI */
    const IID MyIID_IDXGIFactory = {0x7b7166ec, 0x21c7, 0x44ae, {0xb2,0x1a,0xa9,0xae,0x32,0x17,0xd2,0x6e}};
    const IID MyIID_IDXGIAdapter3 = {0x64596741, 0x417e, 0x4ac8, {0x8d,0x28,0x81,0x9b,0xbe,0x4a,0x0d,0x1a}};

    void* factory = nullptr;
    if (FAILED(pCreateDXGIFactory(MyIID_IDXGIFactory, &factory))) {
        FreeLibrary(hDxgi);
        return 0;
    }

    /* Typecast factory to a simple struct that matches IUnknown + IDXGIFactory */
    struct UnknownVtbl {
        HRESULT (STDMETHODCALLTYPE *QueryInterface)(void* This, REFIID riid, void** ppvObject);
        ULONG (STDMETHODCALLTYPE *AddRef)(void* This);
        ULONG (STDMETHODCALLTYPE *Release)(void* This);
    };
    struct FakeFactoryVtbl {
        UnknownVtbl unknown;
        HRESULT (STDMETHODCALLTYPE *EnumAdapters)(void* This, UINT Adapter, void** ppAdapter);
    };
    struct FakeFactory { FakeFactoryVtbl* lpVtbl; };

    FakeFactory* fFactory = (FakeFactory*)factory;
    void* adapter = nullptr;
    int64_t result_mem = 0;

    if (SUCCEEDED(fFactory->lpVtbl->EnumAdapters(fFactory, 0, &adapter))) {
        struct FakeAdapterVtbl {
            UnknownVtbl unknown;
            HRESULT (STDMETHODCALLTYPE *GetDesc)(void* This, MyDXGI_ADAPTER_DESC* pDesc);
        };
        struct FakeAdapter { FakeAdapterVtbl* lpVtbl; };
        FakeAdapter* fAdapter = (FakeAdapter*)adapter;

        MyDXGI_ADAPTER_DESC desc;
        if (SUCCEEDED(fAdapter->lpVtbl->GetDesc(fAdapter, &desc))) {
            if (available_only) {
                /* IDXGIAdapter3 offers QueryVideoMemoryInfo, but for simplicity
                 * we approximate available VRAM as 85% of dedicated VRAM. */
                result_mem = (int64_t)(desc.DedicatedVideoMemory * 0.85);
            } else {
                result_mem = (int64_t)desc.DedicatedVideoMemory;
            }
        }
        fAdapter->lpVtbl->unknown.Release(fAdapter);
    }

    fFactory->lpVtbl->unknown.Release(fFactory);
    FreeLibrary(hDxgi);
    return result_mem;
}
#endif

int64_t amcoli_sys_get_total_vram(void) {
#ifdef _WIN32
    return query_dxgi_vram(false);
#else
    /* Linux: try standard AMD/Intel sysfs path */
    FILE *f = fopen("/sys/class/drm/card0/device/mem_info_vram_total", "r");
    if (f) {
        int64_t val = 0;
        if (fscanf(f, "%lld", &val) == 1) {
            fclose(f);
            return val;
        }
        fclose(f);
    }
    return 0;
#endif
}

int64_t amcoli_sys_get_available_vram(void) {
#ifdef _WIN32
    return query_dxgi_vram(true);
#else
    /* Linux: try standard AMD/Intel sysfs path */
    FILE *f = fopen("/sys/class/drm/card0/device/mem_info_vram_used", "r");
    if (f) {
        int64_t used = 0;
        if (fscanf(f, "%lld", &used) == 1) {
            fclose(f);
            int64_t total = amcoli_sys_get_total_vram();
            if (total > used) return total - used;
        }
        fclose(f);
    }
    return 0;
#endif
}

void amcoli_sys_get_gpu_name(char *name_out, size_t max_len) {
    if (!name_out || max_len == 0) return;
    strcpy(name_out, "Unknown GPU");

#ifdef _WIN32
    typedef HRESULT (WINAPI *CreateDXGIFactoryFunc)(REFIID, void**);
    HMODULE hDxgi = LoadLibraryA("dxgi.dll");
    if (!hDxgi) return;

    CreateDXGIFactoryFunc pCreateDXGIFactory = 
        (CreateDXGIFactoryFunc)GetProcAddress(hDxgi, "CreateDXGIFactory");
    if (!pCreateDXGIFactory) {
        FreeLibrary(hDxgi);
        return;
    }

    struct MyDXGI_ADAPTER_DESC {
        WCHAR Description[128];
        UINT VendorId;
        UINT DeviceId;
        UINT SubSysId;
        UINT Revision;
        SIZE_T DedicatedVideoMemory;
        SIZE_T DedicatedSystemMemory;
        SIZE_T SharedSystemMemory;
        LUID AdapterLuid;
    };

    const IID MyIID_IDXGIFactory = {0x7b7166ec, 0x21c7, 0x44ae, {0xb2,0x1a,0xa9,0xae,0x32,0x17,0xd2,0x6e}};

    void* factory = nullptr;
    if (FAILED(pCreateDXGIFactory(MyIID_IDXGIFactory, &factory))) {
        FreeLibrary(hDxgi);
        return;
    }

    struct UnknownVtbl {
        HRESULT (STDMETHODCALLTYPE *QueryInterface)(void* This, REFIID riid, void** ppvObject);
        ULONG (STDMETHODCALLTYPE *AddRef)(void* This);
        ULONG (STDMETHODCALLTYPE *Release)(void* This);
    };
    struct FakeFactoryVtbl {
        UnknownVtbl unknown;
        HRESULT (STDMETHODCALLTYPE *EnumAdapters)(void* This, UINT Adapter, void** ppAdapter);
    };
    struct FakeFactory { FakeFactoryVtbl* lpVtbl; };

    FakeFactory* fFactory = (FakeFactory*)factory;
    void* adapter = nullptr;

    if (SUCCEEDED(fFactory->lpVtbl->EnumAdapters(fFactory, 0, &adapter))) {
        struct FakeAdapterVtbl {
            UnknownVtbl unknown;
            HRESULT (STDMETHODCALLTYPE *GetDesc)(void* This, MyDXGI_ADAPTER_DESC* pDesc);
        };
        struct FakeAdapter { FakeAdapterVtbl* lpVtbl; };
        FakeAdapter* fAdapter = (FakeAdapter*)adapter;

        MyDXGI_ADAPTER_DESC desc;
        if (SUCCEEDED(fAdapter->lpVtbl->GetDesc(fAdapter, &desc))) {
            size_t i = 0;
            for (i = 0; i < 127 && desc.Description[i] != L'\0' && i < max_len - 1; i++) {
                name_out[i] = (char)desc.Description[i];
            }
            name_out[i] = '\0';
        }
        fAdapter->lpVtbl->unknown.Release(fAdapter);
    }

    fFactory->lpVtbl->unknown.Release(fFactory);
    FreeLibrary(hDxgi);
#else
    FILE *f = fopen("/sys/class/drm/card0/device/device", "r");
    if (f) {
        unsigned int dev_id = 0;
        if (fscanf(f, "0x%x", &dev_id) == 1) {
            sprintf(name_out, "AMD/Intel DRM Graphics (0x%04x)", dev_id);
            fclose(f);
            return;
        }
        fclose(f);
    }
    strncpy(name_out, "Generic Linux GPU", max_len - 1);
#endif
}

void amcoli_print_recommendations(void) {
    int64_t total_ram = amcoli_sys_get_total_ram();
    int64_t total_vram = amcoli_sys_get_total_vram();

    double ram_gb = (double)total_ram / (1024.0 * 1024.0 * 1024.0);
    double vram_gb = (double)total_vram / (1024.0 * 1024.0 * 1024.0);

    int count = 0;
    const struct amcoli_model_info *registry = amcoli_get_model_registry(&count);

    /* AMcoli streams experts from SSD, so only the resident (dense) part of
     * an MoE needs to fit in RAM. Conservatively treat the full size_gb as
     * the resident requirement — anything that fits comfortably is a safe
     * recommendation; larger models are still viable on fast NVMe. */
    double safe_budget = ram_gb * 0.85;

    const struct amcoli_model_info *best_fit = NULL;  /* largest MoE that fits in RAM */
    const struct amcoli_model_info *best_small = NULL;/* best MoE <= 8 GB file size  */
    const struct amcoli_model_info *best_any = NULL;  /* largest model overall       */
    const struct amcoli_model_info *best_moe = NULL;  /* largest MoE overall         */

    for (int i = 0; i < count; i++) {
        const struct amcoli_model_info *m = &registry[i];
        if (!best_any || m->total_params > best_any->total_params) best_any = m;

        if (m->expert_size_mb > 0.0) {
            if (!best_moe || m->total_params > best_moe->total_params) best_moe = m;
            if (m->size_gb <= safe_budget && (!best_fit || m->total_params > best_fit->total_params)) {
                best_fit = m;
            }
            if (m->size_gb <= 8.0 && (!best_small || m->total_params > best_small->total_params)) {
                best_small = m;
            }
        }
    }

    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║           AMcoli — Model Recommendations Guide           ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Detected System Specs:                                  ║\n");
    fprintf(stderr, "║    System RAM: %6.2f GB                                  ║\n", ram_gb);
    if (total_vram > 0) {
        fprintf(stderr, "║    GPU VRAM:   %6.2f GB                                  ║\n", vram_gb);
    } else {
        fprintf(stderr, "║    GPU VRAM:   Not detected (or disabled)                ║\n");
    }
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Best MoE fit for your RAM:                              ║\n");

    if (best_fit) {
        fprintf(stderr, "║    - %-58s║\n", best_fit->name);
        fprintf(stderr, "║      (alias: %s, %.2f GB file, %.1fB params, ", best_fit->alias, best_fit->size_gb, best_fit->total_params);
        fprintf(stderr, "%.1fB active)\n", best_fit->active_params);
        fprintf(stderr, "║       pull command: amcoli pull %-35s║\n", best_fit->alias);
    } else {
        fprintf(stderr, "║    Your RAM is too low for a comfortable MoE fit;      ║\n");
        fprintf(stderr, "║    even small MoEs will rely on SSD streaming.          ║\n");
    }

    if (best_small && best_small != best_fit) {
        fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║  Lightweight MoE (fits on low-end hardware):           ║\n");
        fprintf(stderr, "║    - %-58s║\n", best_small->name);
        fprintf(stderr, "║      (alias: %s, %.2f GB file)                            ║\n", best_small->alias, best_small->size_gb);
    }

    if (best_moe && best_moe != best_fit) {
        fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║  Largest MoE in registry (best on fast NVMe + big RAM): ║\n");
        fprintf(stderr, "║    - %-58s║\n", best_moe->name);
        fprintf(stderr, "║      (alias: %s, %.2f GB file)                            ║\n", best_moe->alias, best_moe->size_gb);
    }

    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  SSD Storage Recommendation:                             ║\n");
    fprintf(stderr, "║    For optimal on-demand streaming:                      ║\n");
    fprintf(stderr, "║    - NVMe Gen 3 (3,500 MB/s): OK (1 - 3 tokens/sec)      ║\n");
    fprintf(stderr, "║    - NVMe Gen 4 (7,000 MB/s): Good (2 - 5 tokens/sec)    ║\n");
    fprintf(stderr, "║    * SATA SSDs or HDDs are NOT recommended.              ║\n");
    fprintf(stderr, "║  View all 36 models anytime with: amcoli list             ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n\n");
}
