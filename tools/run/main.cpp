/**
 * @file main.cpp
 * @brief AMcoli CLI entry point.
 *
 * Supports commands:
 *   amcoli run   --model <path> [--ram-gb N] [--vram-gb N] [--model-info]
 *   amcoli serve --model <path> [--host H] [--port P]  (TODO: Phase 5)
 *   amcoli bench --model <path> [--output bench.json]   (TODO: Phase 2)
 *   amcoli convert <input> <output>                     (TODO: Phase 5)
 *
 * Current run mode is a simulated chat/cache demo until llama.cpp inference is
 * integrated.
 */

#include "amcoli.h"
#include "amcoli-bench.h"
#include "amcoli-sys-info.h"
#include "amcoli-downloader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <filesystem>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <unistd.h>
#endif

/* ── Argument parsing ────────────────────────────────────────────────── */

struct cli_args {
    const char *command;       /* "run", "serve", "bench", "convert" */
    const char *model_path;
    double      ram_gb;        /* -1 = auto */
    double      vram_gb;       /* -1 = auto, 0 = disabled */
    const char *eviction;      /* "lru", "lfu" */
    int32_t     prefetch_depth;
    bool        model_info;    /* --model-info: just print config and exit */
    bool        stats;         /* --stats: print live stats */
    int32_t     verbosity;
    const char *prompt;        /* --prompt or -p */
    int32_t     n_predict;     /* -n: tokens to generate */
    const char *output_json;   /* --output or -o */
    double      zipf_exponent; /* --zipf */
};

static struct cli_args parse_args(int argc, char **argv) {
    struct cli_args args;
    memset(&args, 0, sizeof(args));
    args.ram_gb  = -1.0;
    args.vram_gb = -1.0;
    args.eviction = "lru";
    args.verbosity = 1;
    args.n_predict = 128;
    args.zipf_exponent = 1.0;

    if (argc < 2) return args;
    args.command = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--ram-gb") == 0) {
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "auto") == 0) {
                    args.ram_gb = -1.0;
                } else {
                    args.ram_gb = atof(argv[i + 1]);
                }
                i++;
            }
        } else if (strcmp(argv[i], "--vram-gb") == 0) {
            if (i + 1 < argc) {
                if (strcmp(argv[i + 1], "auto") == 0) {
                    args.vram_gb = -1.0;
                } else {
                    args.vram_gb = atof(argv[i + 1]);
                }
                i++;
            }
        } else if (strcmp(argv[i], "--eviction") == 0) {
            if (i + 1 < argc) args.eviction = argv[++i];
        } else if (strcmp(argv[i], "--prefetch-depth") == 0) {
            if (i + 1 < argc) args.prefetch_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--model-info") == 0) {
            args.model_info = true;
        } else if (strcmp(argv[i], "--stats") == 0) {
            args.stats = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args.verbosity = 2;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            args.verbosity = 0;
        } else if (strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) args.n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) args.output_json = argv[++i];
        } else if (strcmp(argv[i], "--zipf") == 0) {
            if (i + 1 < argc) args.zipf_exponent = atof(argv[++i]);
        }
    }

    /* Position argument fallback for model name if --model not supplied */
    if (!args.model_path && argc >= 3) {
        if (argv[2][0] != '-') {
            args.model_path = argv[2];
        }
    }

    return args;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "\n"
        "AMcoli v%s — Universal MoE Disk-Streaming Inference Engine\n"
        "\n"
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  run      Run simulated chat/cache demo (real inference TODO)\n"
        "  serve    Start OpenAI-compatible API server (TODO)\n"
        "  bench    Run simulated MoE cache benchmark\n"
        "  pull     Download a model from Hugging Face\n"
        "  recommend Print system specs and recommend compatible MoE models\n"
        "  convert  Convert/re-quantize GGUF models (TODO)\n"
        "\n"
        "Options:\n"
        "  -m, --model <path>       Path to GGUF model file (required)\n"
        "      --ram-gb <N|auto>    RAM budget for expert cache (default: auto)\n"
        "      --vram-gb <N|auto>   VRAM budget for expert cache (default: 0)\n"
        "      --eviction <lru|lfu> Cache eviction policy (default: lru)\n"
        "      --prefetch-depth <N> Layers ahead to prefetch (default: 0)\n"
        "      --model-info         Print MoE config and exit\n"
        "      --stats              Print live statistics\n"
        "  -p, --prompt <text>      Input prompt for future real inference\n"
        "  -n <N>                   Tokens to simulate in bench mode (default: 128)\n"
        "  -o, --output <path>      Output JSON file for benchmark\n"
        "      --zipf <val>         Zipfian exponent for benchmark (default: 1.0)\n"
        "  -v, --verbose            Verbose output\n"
        "  -q, --quiet              Minimal output\n"
        "\n"
        "Examples:\n"
        "  %s run -m mixtral-8x7b-q4.gguf --ram-gb auto --stats\n"
        "  %s run -m model.gguf --model-info\n"
        "  %s serve -m model.gguf --ram-gb 24 --vram-gb 8\n"
        "\n",
        AMCOLI_VERSION_STRING, prog, prog, prog, prog
    );
}

static void print_chat_stats_panel(struct amcoli_context *ctx, const char *model_path) {
    struct amcoli_stats stats = amcoli_get_stats(ctx);
    int32_t vr_used = 0, vr_tot = 0, rm_used = 0, rm_tot = 0;
    amcoli_get_cache_info(ctx, &vr_used, &vr_tot, &rm_used, &rm_tot);

    /* Search registry for expert count/parameters matching this config */
    int count = 0;
    const struct amcoli_model_info *registry = amcoli_get_model_registry(&count);
    const struct amcoli_model_info *match = NULL;
    if (model_path) {
        for (int i = 0; i < count; i++) {
            if (strstr(model_path, registry[i].filename) ||
                strstr(model_path, registry[i].alias)) {
                match = &registry[i];
                break;
            }
        }
    }

    /* Render VRAM bar */
    char vr_bar[11] = "----------";
    if (vr_tot > 0) {
        int fill = (vr_used * 10) / vr_tot;
        for (int i = 0; i < fill && i < 10; i++) vr_bar[i] = '#';
    }

    /* Render RAM bar */
    char rm_bar[11] = "----------";
    if (rm_tot > 0) {
        int fill = (rm_used * 10) / rm_tot;
        for (int i = 0; i < fill && i < 10; i++) rm_bar[i] = '#';
    }

    fprintf(stderr, "\n\033[1;30m┌──────────────────────────────────────────────────────────┐\033[0m\n");
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;33mAMcoli Live Cache & Streaming Performance Monitor\033[0m       \033[1;30m│\033[0m\n");
    fprintf(stderr, "\033[1;30m├──────────────────────────────────────────────────────────┤\033[0m\n");
    
    if (match) {
        fprintf(stderr, "\033[1;30m│\033[0m  \033[1;33mModel Specs:\033[0m %5.1fB params total │ %5.2fB active/token      \033[1;30m│\033[0m\n",
            match->total_params, match->active_params);
        double est_speed = 3500.0 / (32.0 * 2.0 * match->expert_size_mb * 0.15);
        if (match->size_gb < 3.0) {
            fprintf(stderr, "\033[1;30m│\033[0m  \033[1;33mEst. Speed :\033[0m In-RAM (~120 t/s)                           \033[1;30m│\033[0m\n");
        } else {
            fprintf(stderr, "\033[1;30m│\033[0m  \033[1;33mEst. Speed :\033[0m %5.1f tokens/sec (based on SSD limit)       \033[1;30m│\033[0m\n",
                est_speed);
        }
        fprintf(stderr, "\033[1;30m├──────────────────────────────────────────────────────────┤\033[0m\n");
    }

    if (vr_tot > 0) {
        fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mVRAM Cache:\033[0m  [%s]  %3d/%-3d slots (%5.1f%%)   \033[1;30m│\033[0m\n",
            vr_bar, vr_used, vr_tot, vr_tot > 0 ? (double)vr_used * 100.0 / vr_tot : 0.0);
    } else {
        fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mVRAM Cache:\033[0m  [Disabled / Offline]                    \033[1;30m│\033[0m\n");
    }

    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;32mRAM Cache :\033[0m  [%s]  %3d/%-3d slots (%5.1f%%)   \033[1;30m│\033[0m\n",
        rm_bar, rm_used, rm_tot, rm_tot > 0 ? (double)rm_used * 100.0 / rm_tot : 0.0);

    fprintf(stderr, "\033[1;30m├──────────────────────────────────────────────────────────┤\033[0m\n");
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;35mCache Traffic:\033[0m VRAM Hits: %-5lld │ RAM Hits: %-5lld        \033[1;30m│\033[0m\n",
        (long long)stats.cache_hits_vram, (long long)stats.cache_hits_ram);
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;31mDisk Misses  :\033[0m %-11lld │ Hit Rate: \033[1;33m%5.1f%%\033[0m          \033[1;30m│\033[0m\n",
        (long long)stats.cache_misses, stats.cache_hit_rate * 100.0);

    fprintf(stderr, "\033[1;30m├──────────────────────────────────────────────────────────┤\033[0m\n");
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;34mDisk Streamer:\033[0m Read: %6.2f MB │ Latency : %6.2f ms      \033[1;30m│\033[0m\n",
        (double)stats.disk_bytes_read / (1024.0 * 1024.0), stats.disk_wait_ms_avg);
    fprintf(stderr, "\033[1;30m└──────────────────────────────────────────────────────────┘\033[0m\n\n");
}

static void print_welcome_banner(void) {
    fprintf(stderr, "\033[1;30m================================================================================\033[0m\n");
    fprintf(stderr, "\033[1;36m[SYSTEM]: LAUNCHING CORE MODULE...\033[0m\n");
    fprintf(stderr, "\033[1;32m[STATUS]: STABLE\033[0m\n");
    fprintf(stderr, "\033[1;35mGet Ready to feel the future>.................\033[0m\n");
    fprintf(stderr, "\033[1;30m================================================================================\033[0m\n\n");
    fprintf(stderr, "\033[1;31m ▄▄▄▄▄▄▄ ▄▄   ▄▄ \033[1;37m▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ \033[0m\n");
    fprintf(stderr, "\033[1;31m█       █  █▄█  █\033[1;37m       █       █  █ █  █       █\033[0m\n");
    fprintf(stderr, "\033[1;31m█   ▄   █   █   █\033[1;37m       █   ▄   █  █▄█  █   ▄   █\033[0m\n");
    fprintf(stderr, "\033[1;31m█  █▄█  █       █\033[1;37m     ▄▄█  █ █  █       █  █ █  █\033[0m\n");
    fprintf(stderr, "\033[1;31m█       █       █\033[1;37m    █  █  █▄█  █       █  █▄█  █\033[0m\n");
    fprintf(stderr, "\033[1;31m█   ▄   █ ██▄██ █\033[1;37m    █▄▄█       █   ▄   █       █\033[0m\n");
    fprintf(stderr, "\033[1;31m█▄▄█ █▄▄█▄█   █▄█\033[1;37m▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█\033[0m\n\n");
    fprintf(stderr, "\033[1;30m================================================================================\033[0m\n");
    fprintf(stderr, " \033[1;33mDEVELOPED BY AWAIS // GITHUB:\033[0m \033[4;34mhttps://github.com/Awais-17\033[0m\n");
    fprintf(stderr, "\033[1;30m================================================================================\033[0m\n\n");
}

static void print_model_registry_table(void) {
    int64_t total_ram = amcoli_sys_get_total_ram();
    int64_t avail_ram = amcoli_sys_get_available_ram();
    int64_t total_vram = amcoli_sys_get_total_vram();
    int64_t avail_vram = amcoli_sys_get_available_vram();
    
    double total_ram_gb = (double)total_ram / (1024.0 * 1024.0 * 1024.0);
    double avail_ram_gb = (double)avail_ram / (1024.0 * 1024.0 * 1024.0);
    double total_vram_gb = (double)total_vram / (1024.0 * 1024.0 * 1024.0);
    double avail_vram_gb = (double)avail_vram / (1024.0 * 1024.0 * 1024.0);
    
    unsigned int cores = std::thread::hardware_concurrency();
    
    std::error_code ec;
    auto space_info = std::filesystem::space(".", ec);
    double free_gb = 0.0;
    if (!ec) {
        free_gb = (double)space_info.available / (1024.0 * 1024.0 * 1024.0);
    }

    /* Recommendation Logic */
    const char *rec_alias = "qwen-14b";
    const char *rec_name = "Qwen3-30B-A3B-Instruct";
    if (total_ram_gb < 8.0) {
        rec_alias = "qwen-3b";
        rec_name = "Qwen1.5-MoE-A2.7B-Chat";
    } else if (total_ram_gb >= 24.0) {
        rec_alias = "mixtral";
        rec_name = "Mixtral-8x7B-Instruct-v0.1";
    }
    
    fprintf(stderr, "\n\033[1;30m┌──────────────────────────────────────────────────────────┐\033[0m\n");
    fprintf(stderr, "\033[1;30m│\033[0m              \033[1;33mAMcoli — System Specifications\033[0m              \033[1;30m│\033[0m\n");
    fprintf(stderr, "\033[1;30m├──────────────────────────────────────────────────────────┤\033[0m\n");
    
    char cpu_str[64];
    if (cores > 0) {
        sprintf(cpu_str, "%d Cores (AVX2 supported)", cores);
    } else {
        strcpy(cpu_str, "Unknown");
    }
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mCPU Cores\033[0m   : %-42s\033[1;30m│\033[0m\n", cpu_str);

    char ram_str[64];
    sprintf(ram_str, "%5.2f GB total (avail: %5.2f GB)", total_ram_gb, avail_ram_gb);
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mSystem RAM\033[0m  : %-42s\033[1;30m│\033[0m\n", ram_str);

    char gpu_desc[128] = {0};
    amcoli_sys_get_gpu_name(gpu_desc, sizeof(gpu_desc));
    char vram_str[128];
    if (total_vram > 0) {
        sprintf(vram_str, "%s (%.1f GB, %.1f GB avail)", gpu_desc, total_vram_gb, avail_vram_gb);
    } else {
        strcpy(vram_str, "Not detected / Offline");
    }
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mGPU Adapter\033[0m : %-42s\033[1;30m│\033[0m\n", vram_str);

    char disk_str[64];
    if (!ec) {
        sprintf(disk_str, "%5.1f GB free on active drive", free_gb);
    } else {
        strcpy(disk_str, "Error querying free space");
    }
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;36mDisk Space\033[0m  : %-42s\033[1;30m│\033[0m\n", disk_str);

    char rec_str[64];
    sprintf(rec_str, "%s (Recommended)", rec_name);
    fprintf(stderr, "\033[1;30m│\033[0m  \033[1;32mRecommend\033[0m   : %-42s\033[1;30m│\033[0m\n", rec_str);
    fprintf(stderr, "\033[1;30m└──────────────────────────────────────────────────────────┘\033[0m\n");

    int count = 0;
    const struct amcoli_model_info *registry = amcoli_get_model_registry(&count);
    
    fprintf(stderr, "\n┌─────┬──────────────────────────────────────┬──────────────┬──────────────┬──────────────────┬──────────┐\n");
    fprintf(stderr, "│ Idx │ Model Name                           │ Total Params │ Active/Token │ Est. Speed (SSD) │ Status   │\n");
    fprintf(stderr, "├─────┼──────────────────────────────────────┼──────────────┼──────────────┼──────────────────┼──────────┤\n");
    for (int i = 0; i < count; i++) {
        bool is_rec = (strcmp(registry[i].alias, rec_alias) == 0);
        bool exists = amcoli_model_exists(registry[i].alias);
        /* Speed estimate: NVMe Gen 3 (3500 MB/s), 32 layers, top-2 active, 85% cache hit rate (15% disk miss) */
        double est_speed = 0.0;
        char speed_str[32];
        if (registry[i].expert_size_mb == 0.0) {
            if (registry[i].size_gb <= total_ram_gb) {
                sprintf(speed_str, "In-RAM (~35 t/s)");
            } else {
                sprintf(speed_str, "Slower (swap)");
            }
        } else {
            est_speed = 3500.0 / (32.0 * 2.0 * registry[i].expert_size_mb * 0.15);
            if (registry[i].size_gb < 3.0) {
                sprintf(speed_str, "In-RAM (~120 t/s)");
            } else {
                sprintf(speed_str, "%4.1f tokens/sec", est_speed);
            }
        }
        
        if (is_rec) {
            fprintf(stderr, "\033[1;32m│  %d  │ %-36s │    %5.1fB    │    %5.2fB    │ %-16s │ %-8s │\033[0m\n",
                i + 1, registry[i].name, registry[i].total_params, registry[i].active_params, speed_str, exists ? "Cached" : "Online");
        } else {
            fprintf(stderr, "│  %d  │ %-36s │    %5.1fB    │    %5.2fB    │ %-16s │ %-8s │\n",
                i + 1, registry[i].name, registry[i].total_params, registry[i].active_params, speed_str, exists ? "Cached" : "Online");
        }
    }
    fprintf(stderr, "└─────┴──────────────────────────────────────┴──────────────┴──────────────┴──────────────────┴──────────┘\n\n");
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
#ifdef _WIN32
    /* Set console code page to UTF-8 to prevent unicode box characters from corrupting */
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    /* Enable Virtual Terminal processing for ANSI codes on Windows Console */
    HANDLE hOut = GetStdHandle(STD_ERROR_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= 0x0004; // ENABLE_VIRTUAL_TERMINAL_PROCESSING
        SetConsoleMode(hOut, dwMode);
    }
#endif

    print_welcome_banner();
    struct cli_args args = parse_args(argc, argv);

    if (!args.command) {
        args.command = "run";
    }

    if (strcmp(args.command, "run") != 0 &&
        strcmp(args.command, "serve") != 0 &&
        strcmp(args.command, "bench") != 0 &&
        strcmp(args.command, "pull") != 0 &&
        strcmp(args.command, "recommend") != 0 &&
        strcmp(args.command, "convert") != 0)
    {
        fprintf(stderr, "Error: Unknown command '%s'\n", args.command);
        print_usage(argv[0]);
        return 1;
    }

    /* If no model path is specified for the 'run' command, show an interactive model selector */
    static char chosen_alias[128] = {0};
    if (strcmp(args.command, "run") == 0 && !args.model_path) {
        int count = 0;
        const struct amcoli_model_info *registry = amcoli_get_model_registry(&count);
        
        while (true) {
            fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║           AMcoli — Interactive Model Selector            ║\n");
            fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
            print_model_registry_table();
            
            fprintf(stderr, "Enter choice (1-%d) or 'q' to quit: ", count);
            fflush(stderr);
            char choice[64];
            if (!fgets(choice, sizeof(choice), stdin)) {
                return 1;
            }
            
            /* Strip trailing newline */
            size_t choice_len = strlen(choice);
            if (choice_len > 0 && choice[choice_len - 1] == '\n') {
                choice[choice_len - 1] = '\0';
                choice_len--;
            }
            
            if (choice_len == 0) continue;
            
            if (strcmp(choice, "q") == 0 || strcmp(choice, "Q") == 0 || strcmp(choice, "/exit") == 0 || strcmp(choice, "/quit") == 0) {
                return 0;
            }
            
            if (strcmp(choice, "/recommend") == 0 || strcmp(choice, "/recoment") == 0) {
                amcoli_print_recommendations();
                continue;
            }
            
            if (strcmp(choice, "/model") == 0 || strcmp(choice, "/models") == 0) {
                continue;
            }
            
            int idx = atoi(choice) - 1;
            if (idx >= 0 && idx < count) {
                strncpy(chosen_alias, registry[idx].alias, sizeof(chosen_alias) - 1);
                args.model_path = chosen_alias;
                break;
            } else {
                fprintf(stderr, "\n\033[1;31mError: Invalid choice '%s'. Please select a number between 1 and %d, or 'q' to quit.\033[0m\n", choice, count);
            }
        }
    }

    if (!args.model_path && strcmp(args.command, "recommend") != 0) {
        fprintf(stderr, "Error: Model path or alias is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Build AMcoli params from CLI args ────────────────────────── */

    struct amcoli_params params = amcoli_params_default();
    params.disk.prefetch_depth = args.prefetch_depth;
    params.verbosity = args.verbosity;
    params.print_stats_on_exit = args.stats;

    /* RAM budget */
    if (args.ram_gb >= 0) {
        params.cache.ram_budget_bytes =
            (int64_t)(args.ram_gb * 1024.0 * 1024.0 * 1024.0);
    }

    /* VRAM budget */
    if (args.vram_gb >= 0) {
        params.cache.vram_budget_bytes =
            (int64_t)(args.vram_gb * 1024.0 * 1024.0 * 1024.0);
    }

    /* Eviction policy */
    if (strcmp(args.eviction, "lfu") == 0) {
        params.cache.eviction_policy = AMCOLI_EVICT_LFU;
    } else if (strcmp(args.eviction, "slru") == 0) {
        params.cache.eviction_policy = AMCOLI_EVICT_SLRU;
    } else {
        params.cache.eviction_policy = AMCOLI_EVICT_LRU;
    }

    /* ── Execute direct commands ────────────────────────────────── */

    if (strcmp(args.command, "recommend") == 0) {
        amcoli_print_recommendations();
        return 0;
    }

    if (strcmp(args.command, "pull") == 0) {
        if (!args.model_path) {
            fprintf(stderr, "Error: Model alias is required (e.g. amcoli pull qwen-3b)\n");
            return 1;
        }
        bool ok = amcoli_download_model(args.model_path);
        return ok ? 0 : 1;
    }

    /* Resolve model path/downloader if using alias */
    char final_model_path[512] = {0};
    if (args.model_path && amcoli_model_is_registered(args.model_path)) {
        if (!amcoli_model_exists(args.model_path)) {
            fprintf(stderr, "Model '%s' is not found locally.\n", args.model_path);
            fprintf(stderr, "Would you like to download it now? (y/n): ");
            fflush(stderr);
            char confirm = 'n';
            if (scanf(" %c", &confirm) == 1 && (confirm == 'y' || confirm == 'Y')) {
                int c;
                while ((c = getchar()) != '\n' && c != EOF); // Clear stdin
                if (!amcoli_download_model(args.model_path)) {
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: Model must be downloaded to run. Use 'amcoli pull %s'.\n", args.model_path);
                return 1;
            }
        }
        amcoli_get_model_path(args.model_path, final_model_path, sizeof(final_model_path));
        params.disk.model_path = final_model_path;
    } else {
        params.disk.model_path = args.model_path;
    }

    /* ── Create context ───────────────────────────────────────────── */

    int32_t err = 0;
    struct amcoli_context *ctx = amcoli_context_create(&params, &err);

    if (!ctx) {
        fprintf(stderr, "Error: Failed to create AMcoli context: %s\n",
            amcoli_error_string(err));
        return 1;
    }

    /* ── Model info mode ──────────────────────────────────────────── */

    if (args.model_info) {
        /* Config was already printed during init if verbosity >= 1.
         * For --model-info with -q, force a print. */
        if (args.verbosity < 1) {
            const struct amcoli_moe_config *config = amcoli_get_moe_config(ctx);
            amcoli_print_moe_config(config);
        }
        amcoli_context_free(ctx);
        return 0;
    }

    /* ── Interactive inference ────────────────────────────────────── */

    if (strcmp(args.command, "run") == 0) {
        fprintf(stderr, "\n=== AMcoli Simulated Chat / Cache Demo ===\n");
        fprintf(stderr, "  Model:       %s\n", params.disk.model_path);
        fprintf(stderr, "  Status:      Demo only. Real llama.cpp token generation is not wired yet.\n");
        fprintf(stderr, "  Commands:    /exit or /quit (exit), /stats (toggle metrics), /clear (clear screen)\n\n");

        char input[2048];
        bool show_live_stats = args.stats;

        /* Clear any pending newline in stdin */
        fflush(stdin);

        while (true) {
            fprintf(stderr, "\n>>> ");
            fflush(stderr);

            if (!fgets(input, sizeof(input), stdin)) {
                break;
            }

            /* Strip trailing newline */
            size_t len = strlen(input);
            if (len > 0 && input[len - 1] == '\n') {
                input[len - 1] = '\0';
                len--;
            }

            if (len == 0) continue;

            /* Check slash commands */
            if (strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
                break;
            }
            if (strcmp(input, "/stats") == 0) {
                show_live_stats = !show_live_stats;
                fprintf(stderr, "Live statistics: %s\n", show_live_stats ? "ON" : "OFF");
                continue;
            }
            if (strcmp(input, "/clear") == 0) {
#ifdef _WIN32
                system("cls");
#else
                system("clear");
#endif
                continue;
            }
            if (strcmp(input, "/recommend") == 0 || strcmp(input, "/recoment") == 0) {
                amcoli_print_recommendations();
                continue;
            }
            if (strcmp(input, "/model") == 0 || strcmp(input, "/models") == 0) {
                int count = 0;
                const struct amcoli_model_info *registry = amcoli_get_model_registry(&count);
                
                fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║           AMcoli — Model Downloader & Switcher           ║\n");
                fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
                print_model_registry_table();
                
                fprintf(stderr, "Select model to download/switch (1-%d) or 'c': ", count);
                fflush(stderr);
                char model_choice[16];
                if (fgets(model_choice, sizeof(model_choice), stdin)) {
                    if (model_choice[0] == 'c' || model_choice[0] == 'C') {
                        continue;
                    }
                    int idx = atoi(model_choice) - 1;
                    if (idx >= 0 && idx < count) {
                        const char *new_alias = registry[idx].alias;
                        
                        /* Download if missing */
                        if (!amcoli_model_exists(new_alias)) {
                            if (!amcoli_download_model(new_alias)) {
                                continue;
                            }
                        }
                        
                        /* Switch model: re-create context */
                        fprintf(stderr, "Switching active model to %s...\n", registry[idx].name);
                        
                        static char new_path[512];
                        amcoli_get_model_path(new_alias, new_path, sizeof(new_path));
                        
                        /* Free old context */
                        amcoli_context_free(ctx);
                        
                        /* Update params path */
                        params.disk.model_path = new_path;
                        
                        /* Re-create context */
                        int32_t switch_err = 0;
                        ctx = amcoli_context_create(&params, &switch_err);
                        if (!ctx) {
                            fprintf(stderr, "Error: Failed to re-initialize model context: %s\n",
                                amcoli_error_string(switch_err));
                            return 1;
                        }
                        
                        fprintf(stderr, "Successfully switched to %s!\n", registry[idx].name);
                        fprintf(stderr, "  Model:       %s\n\n", params.disk.model_path);
                    } else {
                        fprintf(stderr, "Error: Invalid choice.\n");
                    }
                }
                continue;
            }

            /* Simulate response streaming until llama.cpp inference is integrated. */
            fprintf(stderr, "amcoli: ");
            fflush(stderr);

            const char *response[] = {
                "Based", " on", " the", " active", " Mixture", "-of", "-Experts", " configuration,", 
                " I", " have", " resolved", " the", " routing", " path", " for", " your", " query.",
                "\n\n",
                "During", " this", " forward", " pass,", " the", " router", " selected", " active",
                " experts", " in", " each", " transformer", " layer.", " Since", " they", " were",
                " cached", " in", " memory,", " no", " disk", " read", " penalties", " were",
                " incurred.",
                "\n\n",
                "To", " verify", " this", " live", " behavior,", " you", " can", " toggle", 
                " `/stats`", " to", " inspect", " the", " RAM/VRAM", " cache", " hit", " rates",
                " directly", " in", " the", " terminal."
            };
            int n_words = sizeof(response) / sizeof(response[0]);

            const struct amcoli_moe_config *config = amcoli_get_moe_config(ctx);

            /* Exercise the cache/disk code for each query step so stats show real traffic! */
            if (config && config->n_expert > 0 && config->n_moe_layers > 0) {
                for (int32_t l = 0; l < config->n_moe_layers; l++) {
                    int32_t layer_id = config->moe_layer_ids ?
                        config->moe_layer_ids[l] : l;
                    for (int32_t k = 0; k < config->n_expert_used; k++) {
                        int32_t expert_id = (l + k + (int32_t)len) % config->n_expert;
                        void *data;
                        size_t sz;
                        amcoli_ensure_expert(ctx, layer_id, expert_id, &data, &sz);
                    }
                }
            }

            for (int i = 0; i < n_words; i++) {
                fprintf(stderr, "%s", response[i]);
                fflush(stderr);
#ifdef _WIN32
                Sleep(30);
#else
                usleep(30000);
#endif
            }
            fprintf(stderr, "\n");

            if (show_live_stats) {
                print_chat_stats_panel(ctx, params.disk.model_path);
            }
        }
    } else if (strcmp(args.command, "serve") == 0) {
        fprintf(stderr, "AMcoli: serve command not yet implemented (Phase 5)\n");
    } else if (strcmp(args.command, "bench") == 0) {
        struct amcoli_bench_params bparams = amcoli_bench_params_default();
        bparams.num_tokens = args.n_predict;
        bparams.zipf_exponent = args.zipf_exponent;
        bparams.output_json = args.output_json;
        bparams.quiet = (args.verbosity == 0);
        
        err = amcoli_run_benchmark(ctx, &bparams);
        if (err != AMCOLI_OK) {
            fprintf(stderr, "Error: Benchmark run failed: %s\n", amcoli_error_string(err));
        }
    } else if (strcmp(args.command, "convert") == 0) {
        fprintf(stderr, "AMcoli: convert command not yet implemented (Phase 5)\n");
    }

    amcoli_context_free(ctx);
    return 0;
}
