/**
 * @file amcoli-downloader.cpp
 * @brief Downloader implementation using native curl.
 */

#include "amcoli-downloader.h"
#include "amcoli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <io.h>
    #define access _access
    #define F_OK 0
    #define popen _popen
    #define pclose _pclose
#else
    #include <unistd.h>
    #include <sys/stat.h>
#endif

/* ── Registry ────────────────────────────────────────────────────────── */

static const struct amcoli_model_info g_model_registry[] = {
    {
        "qwen-3b",
        "Qwen1.5-MoE-A2.7B-Chat (Q4_K_M)",
        "https://huggingface.co/RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf/resolve/main/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf",
        "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf",
        8.84,
        14.3,
        2.7,
        190.0
    },
    {
        "qwen-14b",
        "Qwen3-30B-A3B-Instruct (Q4_K_M)",
        "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        17.28,
        30.5,
        3.3,
        180.0
    },
    {
        "qwen-57b",
        "Qwen2-57B-A14B-Instruct (Q4_K_M)",
        "https://huggingface.co/mradermacher/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/Qwen2-57B-A14B-Instruct.Q4_K_M.gguf",
        "Qwen2-57B-A14B-Instruct.Q4_K_M.gguf",
        32.46,
        57.0,
        14.0,
        190.0
    },
    {
        "deepseek-moe-16b",
        "DeepSeek-MoE-16B-Chat (Q4_K_M)",
        "https://huggingface.co/mradermacher/deepseek-moe-16b-chat-GGUF/resolve/main/deepseek-moe-16b-chat.Q4_K_M.gguf",
        "deepseek-moe-16b-chat.Q4_K_M.gguf",
        9.2,
        16.4,
        2.8,
        190.0
    },
    {
        "deepseek-lite",
        "DeepSeek-Coder-V2-Lite-Instruct (Q4)",
        "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        10.4,
        16.4,
        2.4,
        70.0
    },
    {
        "dbrx",
        "DBRX-Instruct (IQ2_XXS)",
        "https://huggingface.co/dranger003/dbrx-instruct-iMat.GGUF/resolve/main/ggml-dbrx-instruct-16x12b-iq2_xxs.gguf",
        "ggml-dbrx-instruct-16x12b-iq2_xxs.gguf",
        32.24,
        132.0,
        36.0,
        190.0
    },
    {
        "mixtral",
        "Mixtral-8x7B-Instruct-v0.1 (Q4_K_M)",
        "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        26.4,
        46.7,
        12.9,
        135.0
    },
    {
        "mixtral-8x22b",
        "Mixtral-8x22B-v0.1 (IQ2_XXS)",
        "https://huggingface.co/hermes42/Mixtral-8x22B-v0.1-GGUF/resolve/main/Mixtral-8x22B-v0.1.IQ2_XXS.gguf",
        "Mixtral-8x22B-v0.1.IQ2_XXS.gguf",
        35.28,
        141.0,
        39.0,
        210.0
    },
    {
        "qwen-coder-30b",
        "Qwen3-Coder-30B-A3B-Instruct (Q4_K_M)",
        "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        17.28,
        30.5,
        3.3,
        180.0
    },
    {
        "phi-3.5-moe",
        "Phi-3.5-MoE-instruct (Q4_K_M)",
        "https://huggingface.co/mradermacher/Phi-3.5-MoE-instruct-GGUF/resolve/main/Phi-3.5-MoE-instruct.Q4_K_M.gguf",
        "Phi-3.5-MoE-instruct.Q4_K_M.gguf",
        23.61,
        42.0,
        6.6,
        200.0
    },
    {
        "phi-4-moe",
        "Phi-4-MoE-2x14B-Instruct (Q4_K_M)",
        "https://huggingface.co/mradermacher/Phi4-MoE-2x14B-Instruct-GGUF/resolve/main/Phi4-MoE-2x14B-Instruct.Q4_K_M.gguf",
        "Phi4-MoE-2x14B-Instruct.Q4_K_M.gguf",
        23.06,
        40.0,
        5.6,
        200.0
    },
    {
        "olmoe-7b",
        "OLMoE-1B-7B-0924-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        3.92,
        7.0,
        1.3,
        55.0
    },
    {
        "granite-3b-a800m",
        "Granite-3.0-3B-A800M-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Granite-3.0-3B-A800M-Instruct-GGUF/resolve/main/granite-3.0-3b-a800m-instruct-Q4_K_M.gguf",
        "granite-3.0-3b-a800m-instruct-Q4_K_M.gguf",
        1.92,
        3.3,
        0.8,
        40.0
    },
    {
        "deepseek-v2-lite",
        "DeepSeek-V2-Lite-Chat (Q4_K_M)",
        "https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf",
        "DeepSeek-V2-Lite.Q4_K_M.gguf",
        9.65,
        16.4,
        2.4,
        70.0
    },
    {
        "jamba-mini-1.5",
        "AI21-Jamba-1.5-Mini (Q4_K_M)",
        "https://huggingface.co/mradermacher/AI21-Jamba-Mini-1.5-GGUF/resolve/main/AI21-Jamba-Mini-1.5.Q4_K_M.gguf",
        "AI21-Jamba-Mini-1.5.Q4_K_M.gguf",
        29.02,
        52.0,
        12.0,
        250.0
    },
    {
        "deepseek-v3.1",
        "DeepSeek-V3.1 (UD-TQ1_0)",
        "https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF/resolve/main/DeepSeek-V3.1-UD-TQ1_0.gguf",
        "DeepSeek-V3.1-UD-TQ1_0.gguf",
        158.8,
        671.0,
        37.0,
        180.0
    },
    {
        "glm-5.1",
        "GLM-5.1 (IQ2_XXS)",
        "https://huggingface.co/unsloth/GLM-5.1-GGUF/resolve/main/UD-IQ2_XXS/GLM-5.1-UD-IQ2_XXS-00001-of-00006.gguf",
        "GLM-5.1-UD-IQ2_XXS-00001-of-00006.gguf",
        200.0,
        754.0,
        40.0,
        280.0
    },
    {
        "glm-5.2",
        "GLM-5.2-Colibri (UD-IQ2_XXS)",
        "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/main/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf",
        "GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf",
        238.0,
        744.0,
        40.0,
        280.0
    },
    {
        "kimi-k2.6",
        "Kimi-K2.6 (UD-Q2_K_XL)",
        "https://huggingface.co/unsloth/Kimi-K2.6-GGUF/resolve/main/UD-Q2_K_XL/Kimi-K2.6-UD-Q2_K_XL-00001-of-00008.gguf",
        "Kimi-K2.6-UD-Q2_K_XL-00001-of-00008.gguf",
        350.0,
        1000.0,
        68.0,
        280.0
    },
    {
        "kimi-k3",
        "Kimi-K3 (UD-Q2_K_XL)",
        "https://huggingface.co/unsloth/Kimi-K3-GGUF/resolve/main/UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf",
        "Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf",
        980.0,
        2800.0,
        120.0,
        320.0
    },
    {
        "kimi-coder-72b",
        "Kimi-Dev-72B-Coder (IQ4_NL)",
        "https://huggingface.co/unsloth/Kimi-Dev-72B-GGUF/resolve/main/Kimi-Dev-72B-IQ4_NL.gguf",
        "Kimi-Dev-72B-IQ4_NL.gguf",
        38.48,
        72.5,
        72.5,
        0.0
    },
    {
        "kimi-coder-135m",
        "Kimi-Coder-135M (Q4_K_M)",
        "https://huggingface.co/mradermacher/kimi-coder-135m-GGUF/resolve/main/kimi-coder-135m.Q4_K_M.gguf",
        "kimi-coder-135m.Q4_K_M.gguf",
        0.1,
        0.135,
        0.135,
        0.0
    },
    {
        "qwen-coder-7b",
        "Qwen2.5-Coder-7B-Instruct (Q4_K_M)",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        4.7,
        7.6,
        7.6,
        0.0
    },
    {
        "qwen-coder-32b",
        "Qwen2.5-Coder-32B-Instruct (Q4_K_M)",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        20.3,
        32.5,
        32.5,
        0.0
    },
    {
        "llama-3.1-8b",
        "Meta-Llama-3.1-8B-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        4.9,
        8.0,
        8.0,
        0.0
    },
    {
        "llama-3.1-70b",
        "Meta-Llama-3.1-70B-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        43.0,
        70.6,
        70.6,
        0.0
    },
    {
        "llama-3.2-1b",
        "Llama-3.2-1B-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        1.2,
        1.2,
        1.2,
        0.0
    },
    {
        "llama-3.2-3b",
        "Llama-3.2-3B-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        2.0,
        3.2,
        3.2,
        0.0
    },
    {
        "gemma-2-2b",
        "Gemma-2-2b-it (Q4_K_M)",
        "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "gemma-2-2b-it-Q4_K_M.gguf",
        1.7,
        2.6,
        2.6,
        0.0
    },
    {
        "gemma-2-9b",
        "Gemma-2-9b-it (Q4_K_M)",
        "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "gemma-2-9b-it-Q4_K_M.gguf",
        5.7,
        9.2,
        9.2,
        0.0
    },
    {
        "gemma-2-27b",
        "Gemma-2-27b-it (Q4_K_M)",
        "https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/resolve/main/gemma-2-27b-it-Q4_K_M.gguf",
        "gemma-2-27b-it-Q4_K_M.gguf",
        17.4,
        27.2,
        27.2,
        0.0
    },
    {
        "qwen-2.5-7b",
        "Qwen2.5-7B-Instruct (Q4_K_M)",
        "https://huggingface.co/mradermacher/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        4.36,
        7.6,
        7.6,
        0.0
    },
    {
        "qwen-2.5-72b",
        "Qwen2.5-72B-Instruct (Q4_K_M)",
        "https://huggingface.co/mradermacher/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct.Q4_K_M.gguf",
        "Qwen2.5-72B-Instruct.Q4_K_M.gguf",
        44.16,
        72.5,
        72.5,
        0.0
    },
    {
        "command-r-plus",
        "Command-R-Plus (IQ3_M)",
        "https://huggingface.co/bartowski/c4ai-command-r-plus-GGUF/resolve/main/c4ai-command-r-plus-IQ3_M.gguf",
        "c4ai-command-r-plus-IQ3_M.gguf",
        44.41,
        104.0,
        104.0,
        0.0
    },
    {
        "phi-3-mini",
        "Phi-3-mini-128k-instruct (Q4_K_M)",
        "https://huggingface.co/QuantFactory/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct.Q4_K_M.gguf",
        "Phi-3-mini-128k-instruct.Q4_K_M.gguf",
        2.23,
        3.8,
        3.8,
        0.0
    },
    {
        "phi-4",
        "Phi-4-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf",
        "phi-4-Q4_K_M.gguf",
        8.5,
        14.7,
        14.7,
        0.0
    }
};

static const int g_model_count = sizeof(g_model_registry) / sizeof(g_model_registry[0]);

/* ── API Implementation ──────────────────────────────────────────────── */

const struct amcoli_model_info *amcoli_get_model_registry(int *count_out) {
    if (count_out) *count_out = g_model_count;
    return g_model_registry;
}

static const struct amcoli_model_info *find_model(const char *alias) {
    for (int i = 0; i < g_model_count; i++) {
        if (strcmp(g_model_registry[i].alias, alias) == 0) {
            return &g_model_registry[i];
        }
    }
    return NULL;
}

bool amcoli_model_is_registered(const char *alias) {
    return find_model(alias) != NULL;
}

void amcoli_get_model_path(const char *alias, char *path_out, size_t max_len) {
    const struct amcoli_model_info *info = find_model(alias);
    if (info) {
        snprintf(path_out, max_len, ".models/%s", info->filename);
    } else {
        /* Fallback: treat alias as direct GGUF file path */
        snprintf(path_out, max_len, "%s", alias);
    }
}

bool amcoli_model_exists(const char *alias) {
    char path[512];
    amcoli_get_model_path(alias, path, sizeof(path));

    /* Check file accessibility */
    return access(path, F_OK) == 0;
}

static bool create_models_directory(void) {
#ifdef _WIN32
    /* Returns non-zero on success, or zero if directory exists / failed */
    BOOL success = CreateDirectoryA(".models", NULL);
    if (!success && GetLastError() != ERROR_ALREADY_EXISTS) {
        return false;
    }
    return true;
#else
    struct stat st = {0};
    if (stat(".models", &st) == -1) {
        if (mkdir(".models", 0755) == -1) {
            return false;
        }
    }
    return true;
#endif
}

bool amcoli_download_model(const char *alias) {
    const struct amcoli_model_info *info = find_model(alias);
    if (!info) {
        fprintf(stderr, "Error: Model '%s' is not in the registry.\n", alias);
        return false;
    }

    if (!create_models_directory()) {
        fprintf(stderr, "Error: Failed to create '.models/' folder.\n");
        return false;
    }

    char dest_path[512];
    amcoli_get_model_path(alias, dest_path, sizeof(dest_path));

    fprintf(stderr, "\nAMcoli: Pulling %s...\n", info->name);
    fprintf(stderr, "        Size: %.2f GB\n", info->size_gb);
    fprintf(stderr, "        Target: %s\n\n", dest_path);

    /* Format curl shell command; -C - resumes an interrupted download */
    char cmd[1024];
#ifdef _WIN32
    /* Windows: use curl.exe with quotes */
    snprintf(cmd, sizeof(cmd), "curl.exe -L -C - --progress-bar -o \"%s\" \"%s\"", dest_path, info->url);
#else
    snprintf(cmd, sizeof(cmd), "curl -L -C - --progress-bar -o \"%s\" \"%s\"", dest_path, info->url);
#endif

    /* Execute download */
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "\nError: Download failed. Check network connection or curl availability.\n");
        return false;
    }

    fprintf(stderr, "\nSuccess: Model downloaded successfully to %s\n\n", dest_path);
    return true;
}

bool amcoli_dry_run(const char *alias) {
    const struct amcoli_model_info *info = find_model(alias);
    if (!info) {
        fprintf(stderr, "Error: Model '%s' is not in the registry.\n", alias);
        return false;
    }

    char dest_path[512];
    amcoli_get_model_path(alias, dest_path, sizeof(dest_path));

    fprintf(stderr, "\nAMcoli: Dry-run — nothing will be downloaded.\n");
    fprintf(stderr, "  Alias   : %s\n", info->alias);
    fprintf(stderr, "  Name    : %s\n", info->name);
    fprintf(stderr, "  Size    : %.2f GB\n", info->size_gb);
    fprintf(stderr, "  Params  : %.1fB total, %.1fB active/token\n",
        info->total_params, info->active_params);
    if (info->expert_size_mb > 0.0) {
        fprintf(stderr, "  Experts : ~%.0f MB each (streamed from SSD)\n", info->expert_size_mb);
    } else {
        fprintf(stderr, "  Experts : dense model (no expert streaming)\n");
    }
    fprintf(stderr, "  Target  : %s\n", dest_path);
    fprintf(stderr, "  URL     : %s\n\n", info->url);
    return true;
}

bool amcoli_check_url(const char *url, int *http_code) {
    if (http_code) *http_code = 0;

    char cmd[1024];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd),
        "curl.exe -s -o NUL -w \"%%{http_code}\" -r 0-0 -L --max-time 45 \"%s\"", url);
#else
    snprintf(cmd, sizeof(cmd),
        "curl -s -o /dev/null -w \"%%{http_code}\" -r 0-0 -L --max-time 45 \"%s\"", url);
#endif

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        return false;
    }
    char buf[32] = {0};
    if (fgets(buf, sizeof(buf), pipe) != NULL) {
        buf[sizeof(buf) - 1] = '\0';
    }
    int ret = pclose(pipe);
    (void)ret;

    int code = atoi(buf);
    if (http_code) *http_code = code;
    return (code == 200 || code == 206);
}

int amcoli_check_registry(void) {
    int ok = 0;
    fprintf(stderr, "\nAMcoli: Verifying %d registry URLs (range GET)...\n\n", g_model_count);
    for (int i = 0; i < g_model_count; i++) {
        int code = 0;
        bool reachable = amcoli_check_url(g_model_registry[i].url, &code);
        if (reachable) ok++;

        fprintf(stderr, "  %-18s %-42s %s (%d)\n",
            g_model_registry[i].alias,
            g_model_registry[i].name,
            reachable ? "\033[1;32mOK\033[0m" : "\033[1;31mBROKEN\033[0m",
            code);
    }
    fprintf(stderr, "\n  Result: %d / %d reachable\n\n", ok, g_model_count);
    return ok;
}

/* ── Update check (once/day via a %TEMP% stamp) ──────────────────────── */

#define AMCOLI_UPDATE_URL "https://raw.githubusercontent.com/Awais-17/AMcoli/main/VERSION"

static int amcoli_parse_version(const char *s, int out[3]) {
    out[0] = out[1] = out[2] = 0;
    int n = sscanf(s, "%d.%d.%d", &out[0], &out[1], &out[2]);
    if (n < 3) return -1;
    /* Reject trailing junk (e.g. "404: Not Found" parses 404 but is invalid) */
    const char *p = s;
    int dots = 0;
    while (*p) {
        if (*p >= '0' && *p <= '9') { p++; continue; }
        if (*p == '.' && dots < 2) { dots++; p++; continue; }
        return -1;
    }
    return (dots == 2) ? 0 : -1;
}

void amcoli_check_for_update(void) {
    /* Resolve the stamp path once per call */
    char stamp_path[512];
    char today[32] = {0};

#ifdef _WIN32
    const char *tmp = getenv("TEMP");
    if (!tmp) tmp = ".";
    snprintf(stamp_path, sizeof(stamp_path), "%s\\amcoli-update-stamp", tmp);
#else
    const char *tmp = getenv("TMPDIR");
    if (!tmp) tmp = "/tmp";
    snprintf(stamp_path, sizeof(stamp_path), "%s/amcoli-update-stamp", tmp);
#endif

    /* Skip if we already checked today */
    time_t now = time(NULL);
    if (now != (time_t)-1) {
        struct tm tmv;
#ifdef _WIN32
        localtime_s(&tmv, &now);
#else
        localtime_r(&now, &tmv);
#endif
        strftime(today, sizeof(today), "%Y-%m-%d", &tmv);

        char stamp[32] = {0};
        FILE *in = fopen(stamp_path, "r");
        if (in) {
            if (fgets(stamp, sizeof(stamp), in) != NULL) {
                char *nl = strpbrk(stamp, "\r\n");
                if (nl) *nl = '\0';
            }
            fclose(in);
            if (strcmp(stamp, today) == 0) {
                return;
            }
        }
    }

    /* Fetch the remote VERSION file (fast, never blocks long) */
    char cmd[768];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd),
        "curl.exe -s --max-time 8 --connect-timeout 3 \"%s\"", AMCOLI_UPDATE_URL);
#else
    snprintf(cmd, sizeof(cmd),
        "curl -s --max-time 8 --connect-timeout 3 \"%s\"", AMCOLI_UPDATE_URL);
#endif

    char remote[64] = {0};
    FILE *pipe = popen(cmd, "r");
    if (pipe) {
        if (fgets(remote, sizeof(remote), pipe) != NULL) {
            remote[sizeof(remote) - 1] = '\0';
        }
        (void)pclose(pipe);
    }

    char *nl = strpbrk(remote, "\r\n");
    if (nl) *nl = '\0';

    int rv[3], lv[3];
    bool have_remote = (remote[0] != '\0') && amcoli_parse_version(remote, rv) == 0;
    bool have_local  = amcoli_parse_version(AMCOLI_VERSION_STRING, lv) == 0;

    /* Always write the stamp so the check runs at most once per day, even
     * when offline (keeps startup network traffic bounded). */
    FILE *stamp = fopen(stamp_path, "w");
    if (stamp) {
        fputs(today, stamp);
        fclose(stamp);
    }

    if (!have_remote || !have_local) {
        return;
    }

    bool newer = false;
    if (rv[0] != lv[0])      newer = rv[0] > lv[0];
    else if (rv[1] != lv[1]) newer = rv[1] > lv[1];
    else                     newer = rv[2] > lv[2];

    if (newer) {
        fprintf(stderr, "\033[1;33m[AMcoli]\033[0m A new version (v%s) is available — you have v%s. "
                        "Download: https://github.com/Awais-17/AMcoli/releases\n",
                        remote, AMCOLI_VERSION_STRING);
    }
}
