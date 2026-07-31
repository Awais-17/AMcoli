# AMcoli Progress Log

This file is a durable handoff log for future agents. Keep it updated whenever
you inspect, change, test, or defer work.

## Goal

Turn AMcoli from a MoE disk-streaming prototype/simulator into an honest,
working local inference runtime capable of running GGUF models, then add
disk-streamed MoE expert loading on top.

## Current Snapshot

- Project: C++17/CMake, MSVC "Visual Studio 16 2019" Release config.
- Core modules exist and are linked: `src/llama-disk-streamer.cpp` (GGUF mmap +
  expert offset index + async prefetch), `src/llama-moe-cache.cpp` (two-tier
  LRU/LFU/SLRU cache), `src/llama-moe-router.cpp` (GGUF metadata → MoE config),
  `src/amcoli-downloader.cpp` (36-entry model registry + curl pull),
  `src/amcoli-sys-info.cpp` (RAM/VRAM/CPU/GPU/Disk + recommendations).
- Real llama.cpp inference is wired into the `run` chat loop (greedy decoding,
  `llama_token_to_piece` streaming). Version b3400 via FetchContent.
- The CLI exposes `run`, `pull`, `list`, `check`, `info`, `version`, `bench`,
  `recommend`, and placeholders for `serve`/`convert`.
- The "VRAM" tier is currently CPU memory pretending to be a faster tier (real
  ggml GPU buffers are a roadmap item).

## Completed In This Session (2026-08-01)

- Re-audited all 36 registry URLs with `curl -r 0-0 -L` range GETs.
- Replaced 12 broken registry entries with verified sources (RichardErkhov,
  unsloth, mradermacher, bartowski, dranger003, hermes42, QuantFactory) —
  filenames, actual `size_gb`, and quant in display names all updated.
- Removed 6 dead entries (bartowski 401-gated, no single-file source):
  `deepseek-v3`, `jetmoe`, `nemotron-base`, `nemotron-instruct`,
  `nemotron-reward`, `grok-1`. Registry count is now 36 and derived from
  `sizeof()`.
- Added to `src/amcoli-downloader.cpp/.h`: `amcoli_dry_run(alias)`,
  `amcoli_check_url(url, &code)`, `amcoli_check_registry()`; `_popen`/`_pclose`
  defines on Win32.
- Made `pull` resumable: curl now uses `-C -` + `--progress-bar`.
- Added CLI commands: `list`, `check`, `info <alias>`, `version`, `help`; bare
  `--version`/`--help` in argv[1] handled; `pull --dry-run` flag.
- Added chat slash commands `/help`, `/?`, `/version`, `/memory`, `/stats`,
  `/clear`; selector loop got `/help` and `/version`.
- Made `recommend` and the table highlight data-driven: pick largest MoE with
  `size_gb <= 0.85 × total_ram` (`amcoli_print_recommendations()` in
  `src/amcoli-sys-info.cpp`).
- Fixed usage text missing prog arg; fixed C4267 warnings with `(int32_t)`
  casts at main.cpp:778/803 (incl. `strlen` cast for `llama_tokenize`).
- Rewrote `CONTEXT.md` with AMcoli branding + full agent guide (layout, CLI
  commands, registry conventions, build/test/deploy, milestones, conventions).
- Rebuilt clean (no warnings), all tests pass (bench 1, disk-streamer 7,
  moe-cache 10, router-config 7), verified CLI smoke tests (version,
  `--version`, list 36 rows, check 36/36, info, pull --dry-run, recommend,
  selector + chat slash commands).
- Redeployed `amcoli.exe` (94,208 bytes) to all 3 deploy targets and verified
  copies.

### Version 0.1.1 + Release Packaging

- Bumped version 0.1.0 → 0.1.1 in `include/amcoli.h` (macros + string),
  `CMakeLists.txt` (`project(VERSION ...)`), `npm-wrapper/package.json`.
- Added a repo-root `VERSION` file (content `0.1.1`) as the single source for
  the update check (also placed inside the release zip).
- Added `amcoli_check_for_update()` in `src/amcoli-downloader.cpp/.h`: fetches
  `https://raw.githubusercontent.com/Awais-17/AMcoli/main/VERSION` via curl
  (max 8s), strictly parses semver (`amcoli_parse_version` rejects trailing
  junk like "404: Not Found"), compares to `AMCOLI_VERSION_STRING`, prints a
  one-line notice when remote is newer. Gated to once per day via a stamp file
  `%TEMP%\amcoli-update-stamp` (written even on failure to bound startup
  network traffic).
- Wired the check into `tools/run/main.cpp` as a detached `std::thread`, launched
  only for long-running commands (`run`, `pull`) so the thread finishes before
  process exit (short commands would kill the thread).
- Removed unused internal readers `read_u8`/`read_i32`/`read_f32` from
  `src/llama-disk-streamer.cpp` to silence MSVC C4505 warnings.
- Built release zip `build/Release/amcoli-v0.1.1-win64.zip` bundling
  `amcoli.exe` + `ggml.dll` + `llama.dll` + `VERSION` (833,595 bytes);
  verified it extracts and runs standalone.
- Redeployed `amcoli.exe` (96,768 bytes) to all 3 deploy targets after killing
  an orphan process that was locking two of them.

## Verification So Far

- `cmake --build build --config Release` succeeds with zero warnings
  (MSVC `/W4 /wd4100 /wd4996`).
- Tests: `test-bench.exe` 1 passed; `test-disk-streamer.exe` 7 passed;
  `test-moe-cache.exe` 10 passed; `test-router-config.exe` 7 passed.
- `amcoli check` reports 36/36 registry URLs reachable.
- Note: `.models/` currently contains only stubs (29-byte and 42 MB files), not
  real GGUF downloads, so full `run` inference cannot be smoke-tested offline.

## Suggested Next Command Sequence

From a normal Windows Developer PowerShell in the repo:

```powershell
cmake -B build -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\test-moe-cache.exe
.\build\Release\test-router-config.exe
.\build\Release\test-disk-streamer.exe
.\build\Release\test-bench.exe
.\build\Release\amcoli.exe check
```

Redeploy after any rebuild to the 3 targets in `CONTEXT.md` §5 (kill any
lingering `amcoli.exe` first — an orphan was locking the deploy files this
session).

## High-Priority Work

1. Implement real asynchronous speculative prefetch using top-$k$ router-logit
   history (roadmap §7).
2. True VRAM buffers via ggml backend — the VRAM cache tier currently fakes it.
3. Implement `serve` (OpenAI-style HTTP) and `convert` placeholders (Phase 5).
4. Keep CLI wording honest: any simulated path should say so.

## Notes For Future Agents

- Registry conventions are documented in `CONTEXT.md` §4: register multi-shard
  models by first shard, never `.gguf.partNof8` or mid-shards, keep
  `expert_size_mb = 0.0` for dense models, keep `size_gb` as `double`.
- Engine conventions in `CONTEXT.md` §8: C++17/MSVC, no warnings tolerated, no
  comments unless asked, preserve the public C API in `include/amcoli.h`.
- When adding a CLI command: extend `parse_args`/dispatch in
  `tools/run/main.cpp` AND the usage text, AND test the argv[1] bare-flag path.
- When bumping the version, update `include/amcoli.h:31-34`,
  `CMakeLists.txt` `project(VERSION ...)`, `npm-wrapper/package.json`, AND the
  repo-root `VERSION` file together (see `CONTEXT.md` §8).
- Avoid committing generated build outputs, `.models/`, or GGUF model files.
