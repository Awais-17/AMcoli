/**
 * @file amcoli-downloader.cpp
 * @brief Downloader implementation using native curl.
 */

#include "amcoli-downloader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <io.h>
    #define access _access
    #define F_OK 0
#else
    #include <unistd.h>
    #include <sys/stat.h>
#endif

/* ── Registry ────────────────────────────────────────────────────────── */

static const struct amcoli_model_info g_model_registry[] = {
    {
        "qwen-3b",
        "Qwen1.5-MoE-A2.7B-Chat (Q4_K_M)",
        "https://huggingface.co/tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf",
        "Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf",
        8.2,
        14.3,
        2.7,
        190.0
    },
    {
        "qwen-14b",
        "Qwen3-30B-A3B-Instruct (Q4_K_M)",
        "https://huggingface.co/byteshape/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507.Q4_K_M.gguf",
        "Qwen3-30B-A3B-Instruct-2507.Q4_K_M.gguf",
        18.2,
        30.5,
        3.3,
        180.0
    },
    {
        "qwen-57b",
        "Qwen2-57B-A14B-Instruct (Q4_K_M)",
        "https://huggingface.co/bartowski/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/Qwen2-57B-A14B-Instruct-Q4_K_M.gguf",
        "Qwen2-57B-A14B-Instruct-Q4_K_M.gguf",
        34.2,
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
        "deepseek-v3",
        "DeepSeek-V3 (IQ2_XXS)",
        "https://huggingface.co/bartowski/DeepSeek-V3-GGUF/resolve/main/DeepSeek-V3-IQ2_XXS.gguf",
        "DeepSeek-V3-IQ2_XXS.gguf",
        23.8,
        671.0,
        37.0,
        180.0
    },
    {
        "jetmoe",
        "JetMoe-8B-Chat (Q4_K_M)",
        "https://huggingface.co/bartowski/JetMoe-8B-Chat-GGUF/resolve/main/JetMoe-8B-Chat-Q4_K_M.gguf",
        "JetMoe-8B-Chat-Q4_K_M.gguf",
        4.8,
        8.0,
        2.2,
        55.0
    },
    {
        "dbrx",
        "DBRX-Instruct (IQ2_XXS)",
        "https://huggingface.co/bartowski/dbrx-instruct-GGUF/resolve/main/dbrx-instruct-IQ2_XXS.gguf",
        "dbrx-instruct-IQ2_XXS.gguf",
        46.2,
        132.0,
        36.0,
        190.0
    },
    {
        "nemotron-base",
        "Nemotron-4-340B-Base (IQ2_XXS)",
        "https://huggingface.co/bartowski/Nemotron-4-340B-Base-GGUF/resolve/main/Nemotron-4-340B-Base-IQ2_XXS.gguf",
        "Nemotron-4-340B-Base-IQ2_XXS.gguf",
        96.5,
        340.0,
        68.0,
        280.0
    },
    {
        "nemotron-instruct",
        "Nemotron-4-340B-Instruct (IQ2_XXS)",
        "https://huggingface.co/bartowski/Nemotron-4-340B-Instruct-GGUF/resolve/main/Nemotron-4-340B-Instruct-IQ2_XXS.gguf",
        "Nemotron-4-340B-Instruct-IQ2_XXS.gguf",
        96.5,
        340.0,
        68.0,
        280.0
    },
    {
        "nemotron-reward",
        "Nemotron-4-340B-Reward (IQ2_XXS)",
        "https://huggingface.co/bartowski/Nemotron-4-340B-Reward-GGUF/resolve/main/Nemotron-4-340B-Reward-IQ2_XXS.gguf",
        "Nemotron-4-340B-Reward-IQ2_XXS.gguf",
        96.5,
        340.0,
        68.0,
        280.0
    },
    {
        "grok-1",
        "Grok-1 (IQ2_XXS)",
        "https://huggingface.co/bartowski/Grok-1-GGUF/resolve/main/Grok-1-IQ2_XXS.gguf",
        "Grok-1-IQ2_XXS.gguf",
        110.3,
        314.0,
        86.0,
        310.0
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
        "https://huggingface.co/bartowski/Mixtral-8x22B-v0.1-GGUF/resolve/main/Mixtral-8x22B-v0.1-IQ2_XXS.gguf",
        "Mixtral-8x22B-v0.1-IQ2_XXS.gguf",
        45.1,
        141.0,
        39.0,
        210.0
    },
    {
        "glm-5.1",
        "GLM-5.1 (IQ2_XXS)",
        "https://huggingface.co/unsloth/GLM-5.1-GGUF/resolve/main/GLM-5.1-IQ2_XXS.gguf",
        "GLM-5.1-IQ2_XXS.gguf",
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
        "https://huggingface.co/unsloth/Kimi-K3-GGUF/resolve/main/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00016.gguf",
        "Kimi-K3-UD-Q2_K_XL-00001-of-00016.gguf",
        980.0,
        2800.0,
        120.0,
        320.0
    },
    {
        "kimi-coder-72b",
        "Kimi-Dev-72B-Coder (Q4_K_M)",
        "https://huggingface.co/unsloth/Kimi-Dev-72B-GGUF/resolve/main/Kimi-Dev-72B-Q4_K_M.gguf",
        "Kimi-Dev-72B-Q4_K_M.gguf",
        42.5,
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

    /* Format curl shell command */
    char cmd[1024];
#ifdef _WIN32
    /* Windows: use curl.exe with quotes */
    snprintf(cmd, sizeof(cmd), "curl.exe -L --progress-bar -o \"%s\" \"%s\"", dest_path, info->url);
#else
    snprintf(cmd, sizeof(cmd), "curl -L --progress-bar -o \"%s\" \"%s\"", dest_path, info->url);
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
