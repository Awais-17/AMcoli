# AMcoli — Agent Context & Developer Guide

This document is the authoritative context for AI coding assistants (and
humans) working on AMcoli. Read it fully before modifying the codebase.

---

## 0. Branding (shown on every CLI start)

```
[SYSTEM]: LAUNCHING CORE MODULE...
[STATUS]: STABLE
Get Ready to feel the future>.................

 ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄
█       █  █▄█  █       █       █  █ █  █       █
█   ▄   █   █   █       █   ▄   █  █▄█  █   ▄   █
█  █▄█  █       █     ▄▄█  █ █  █       █  █ █  █
█       █       █    █  █  █▄█  █       █  █▄█  █
█   ▄   █ ██▄██ █    █▄▄█       █   ▄   █       █
█▄▄█ █▄▄█▄█   █▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█

================================================================================
 DEVELOPED BY AWAIS // GITHUB: https://github.com/Awais-17
================================================================================
```

The banner lives in `print_welcome_banner()` in `tools/run/main.cpp:231`.

---

## 1. Project Motive

Mixture-of-Experts (MoE) models activate only a fraction of parameters per
token (e.g., 2.7B of 14B, or 37B of 671B), but standard runtimes still load
the *entire* weight file (tens–hundreds of GB) into RAM/VRAM, making them
impossible on consumer hardware (16GB RAM laptops, 4GB GPUs).

**AMcoli** solves this by treating NVMe SSD, RAM, and VRAM as one unified
memory hierarchy: only non-MoE weights (attention, embeddings, shared experts)
stay resident, while expert layers stream from SSD on demand through a
two-tier cache (VRAM + RAM) with async prefetch.

---

## 2. Repository Layout

```
AMcoli/
├── CMakeLists.txt          # MSVC "Visual Studio 16 2019" Release config
├── include/amcoli.h        # Public C API (context, cache, stats, version)
├── src/
│   ├── amcoli-downloader.cpp/.h   # Model registry (36 entries) + curl pull + URL health check
│   ├── amcoli-sys-info.cpp/.h     # RAM/VRAM/CPU/GPU/Disk detection + recommendations
│   ├── llama-disk-streamer.cpp    # GGUF mmap + expert offset index + async prefetch
│   ├── llama-moe-cache.cpp        # Two-tier LRU/LFU/SLRU expert cache
│   ├── llama-moe-router.cpp       # GGUF metadata → MoE config extraction
│   └── amcoli-bench.cpp/.h        # Zipf-access benchmark harness
├── tools/run/main.cpp     # CLI entry point (banner, selector, chat, commands)
├── tests/
│   ├── test-disk-streamer.cpp     # 7 tests
│   ├── test-moe-cache.cpp         # 10 tests
│   ├── test-router-config.cpp     # 7 tests
│   └── test-bench.cpp             # 1 test
├── npm-wrapper/            # @github/amcoli npm package (bin/index.js + scripts/install.js)
└── scripts/install.ps1     # PowerShell installer (GitHub release asset → bin/)
```

---

## 3. Core Architecture

1. **`src/llama-disk-streamer.cpp`** — Opens/mmaps the GGUF file, parses tensor
   layout, builds an offset index of `(layer_id × expert_id)` → byte ranges, and
   services `amcoli_ensure_expert()` reads with `madvise`/prefetch hints.
2. **`src/llama-moe-cache.cpp`** — Fixed-slot two-tier cache (VRAM + RAM).
   Insertion, LRU/LFU/SLRU eviction, dedup, stress-tested (10 tests).
3. **`src/llama-moe-router.cpp`** — Extracts `n_expert`, `n_expert_used`,
   `n_moe_layers`, `moe_layer_ids[]`, and per-expert byte sizes from GGUF
   metadata (works across architectures: qwen_moe, dbrx, jetmoe, mixtral, etc.).
4. **`src/amcoli-downloader.cpp`** — Static model registry mapping short aliases
   (`qwen-3b`, `mixtral`, `deepseek-v3.1`, …) to Hugging Face `resolve/main`
   URLs; downloads via system `curl.exe` with a `--progress-bar`.
5. **`tools/run/main.cpp`** — UTF-8 console setup (`SetConsoleOutputCP(65001)` +
   VT processing), welcome banner, interactive model selector loop, llama.cpp
   inference loop, and the `pull`/`list`/`check`/`info`/`version`/`bench`/
   `recommend` commands.
6. **`npm-wrapper/lib/agent.js`** — The `amcoli agent` interface (Node, runs via
   the npm wrapper, not the native exe). A local AI coding assistant with real
   tools (`list_dir`, `read_file` w/ line ranges + size guard, `search_files`
   w/ regex, `get_system_info`, `write_file`, `edit_file`, `run_command`,
   `run_amcoli`), approval flow (`--yes`, `--auto`, per-call prompt), JSONL
   history in `~/.amcoli/agent-history.jsonl`, HTTPS + Bearer API-key support,
   optional SSE streaming, and `/model`/`/api`/`/clear`/`/reset` slash commands.
   Defaults to local Ollama (`http://127.0.0.1:11434/v1`, `qwen-coder-32b`);
   overridable via CLI flags or `AMCOLI_API_URL`/`AMCOLI_MODEL`/`AMCOLI_API_KEY`
   env vars.

### CLI commands
| Command       | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| `amcoli`      | Interactive model selector → download → chat                   |
| `amcoli run`  | Chat loop (`/help`, `/stats`, `/memory`, `/clear`, `/model`…)  |
| `amcoli agent`| Agentic coding assistant (Node/npm wrapper; real filesystem + shell tools) |
| `amcoli pull <alias>` | Download a model from the registry (resumable)          |
| `amcoli list` | Print the model registry table without entering the selector   |
| `amcoli check`| Verify all 36 registry URLs are reachable (curl range GET)     |
| `amcoli info <alias>` | Show details + download URL for one model            |
| `amcoli version` / `--version` | Print version string                             |
| `amcoli recommend` | Print hardware-based model recommendation              |
| `amcoli bench` | Run the Zipf benchmark over the expert cache                  |
| `amcoli serve` / `convert` | Placeholders (Phase 5, not implemented)         |

---

## 4. Model Registry (36 entries)

`src/amcoli-downloader.cpp` starts at line 25. Schema:

```c
struct amcoli_model_info {
    const char *alias;          /* "qwen-3b" */
    const char *name;           /* "Qwen1.5-MoE-A2.7B-Chat (Q4_K_M)" */
    const char *url;            /* https://huggingface.co/<org>/<repo>/resolve/main/<file> */
    const char *filename;       /* local .models/<file> */
    double      size_gb;
    double      total_params;   /* billions */
    double      active_params;  /* billions */
    double      expert_size_mb; /* 0.0 = dense model (no expert streaming) */
};
```

`g_model_count` is derived from `sizeof()` — always 36 after edits. `main.cpp`
reads the registry via `amcoli_get_model_registry(&count)` (dynamic; no
hardcoded table size).

**Registry conventions (verified 2026-08-01):**
- Multi-shard models are registered by *first shard*: `UD-IQ2_XXS/...-00001-of-000NN.gguf`
  (matches unsloth GLM-5.1/GLM-5.2/Kimi-K3 layout).
- Models whose `expert_size_mb == 0.0` are dense — they render as
  "In-RAM (~35 t/s)" / "Slower (swap)" in the table.
- Do **not** register `.gguf.partNof8` files or `0000N-of-` mid-shards; the
  downloader fetches exactly one file.
- Registry URLs were re-audited today; a `amcoli check` run must report
  36/36 OK. The canonical registry check method is a `curl -r 0-0 -L` range
  GET expecting HTTP 206/200.

---

## 5. Build & Test

```powershell
# Configure (once)
cmake -B build -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Run all tests
.\build\Release\test-disk-streamer.exe
.\build\Release\test-moe-cache.exe
.\build\Release\test-router-config.exe
.\build\Release\test-bench.exe

# Run CLI
.\build\Release\amcoli.exe
```

**Outputs:** `build\Release\amcoli.exe`, `build\Release\test-*.exe`,
`build\bin\Release\ggml.dll`, `build\bin\Release\llama.dll`
(llama.cpp is fetched via `FetchContent` at tag `b3400`).

**Deploy targets** (run `amcoli` from any shell → these three copies):
1. `C:\Users\mdawa\.amcoli\bin\amcoli.exe`
2. `C:\Users\mdawa\AppData\Roaming\npm\node_modules\@github\amcoli\bin\amcoli.exe`
3. `npm-wrapper\bin\amcoli.exe`

---

## 6. Achievements & Milestones

- **GGUF magic bug fixed** (`0x46475547` → `0x46554747`) in
  `src/llama-disk-streamer.cpp:73` and `src/llama-moe-router.cpp:28`, plus the
  4 synthetic-GGUF writers. All tests pass.
- **Dynamic hardware auto-detection**: CPU cores, RAM, disk free space, and
  GPU via DXGI (e.g. "RTX 3050 Laptop GPU").
- **Hardware-aware recommendation** (green-highlighted row in the table).
- **Robust input selector loop** — handles invalid input without crashing.
- **36-model registry** with fully URL-audited GGUF entries (Qwen, DeepSeek,
  GLM-5.1/5.2, Kimi-K2.6/K3, Phi-MoE, OLMoE, Granite, Jamba, Mixtral, etc.).
- **UTF-8 console** fix via `SetConsoleOutputCP(65001)`.
- **Carriage-return progress bars** via curl `--progress-bar`.
- **Real llama.cpp inference** wired into the `run` chat loop (greedy
  decoding, `llama_token_to_piece` streaming).

---

## 7. Roadmap / Current Work

- **In-progress**: None currently blocked; feature upgrades are ongoing.
- **Planned**:
  - Speculative prefetch using top-$k$ router-logit history to fetch
    next-layer experts asynchronously before the active layer finishes.
  - `serve` (OpenAI-style HTTP endpoint) and `convert` commands (Phase 5).
  - True VRAM buffers via ggml backend (the "VRAM" tier is currently CPU
    memory pretending to be faster).

- **Done (2026-08-01)**: Registry audit (36/36 verified, 12 repointed, 6 dead
  removed), `list`/`check`/`info`/`version`/`help` commands, chat/selector
  slash commands, data-driven `recommend`, resumable `pull` (`-C -`), and
  version 0.1.1 with a once/day update check. The update check fetches
  `https://raw.githubusercontent.com/Awais-17/AMcoli/main/VERSION` via curl in
  a background thread (long-running commands only), gated by a stamp in
  `%TEMP%\amcoli-update-stamp`, and prints a notice when the remote semver is
  newer. A release zip (`build/Release/amcoli-v0.1.1-win64.zip`) bundles
  `amcoli.exe` + `ggml.dll` + `llama.dll` + `VERSION`.

---

## 8. Engineering Conventions (MUST follow)

- **C++17**, C11. MSVC `/W4 /wd4100 /wd4996`, no warnings tolerated.
- **No comments unless asked** — but keep existing block comments in style.
- Preserve the public C API in `include/amcoli.h`; export C functions with
  `extern "C"`.
- Registry edits: keep alphabetic-ish grouping, keep `expert_size_mb = 0.0`
  for dense models, keep sizes in GB as `double`.
- When adding a CLI command: extend `parse_args`/dispatch in
  `tools/run/main.cpp` AND the command list in the usage text.
- When changing the banner/version: bump `AMCOLI_VERSION_*` in
  `include/amcoli.h:31-34`, `CMakeLists.txt` `project(VERSION ...)`,
  `npm-wrapper/package.json`, AND the repo-root `VERSION` file together.
  The `VERSION` file is the canonical remote source for the once/day update
  check; it must always match `AMCOLI_VERSION_STRING`.
- Never commit build outputs, `.models/`, or GGUF files (see `.gitignore`).
- After any change: rebuild + run all 4 test binaries + redeploy to the 3
  deploy targets listed in §5.
