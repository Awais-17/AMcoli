# AMcoli Progress Log

This file is a durable handoff log for future agents. Keep it updated whenever
you inspect, change, test, or defer work.

## Goal

Turn AMcoli from a MoE disk-streaming prototype/simulator into an honest,
working local inference runtime capable of running GGUF models, then add
disk-streamed MoE expert loading on top.

## Current Snapshot

- Project: C++17/CMake.
- Core modules exist for GGUF metadata parsing, expert offset indexing,
  fixed-slot cache, prefetch hints, system info, downloader registry, and a
  simulated benchmark.
- The CLI chat path currently prints a canned/demo response and only exercises
  cache reads for statistics.
- The project does not currently link llama.cpp or perform real token
  generation.
- The "VRAM" tier is currently CPU memory pretending to be a faster tier.

## Completed In This Session

- Created this handoff log.
- Added `.gitignore` entries for generated build output, local model files,
  GGUF files, and benchmark JSON.
- Updated public docs/header/CLI wording so the current run mode is clearly a
  simulated chat/cache demo, not real llama.cpp inference.
- Fixed fallback MoE layer ID population in `moe_router_extract_config()`.
- Changed context initialization to build the expert index over total
  transformer layers when available, so sparse MoE layer IDs can be addressed.
- Updated simulated CLI and benchmark paths to call `amcoli_ensure_expert()`
  with actual `moe_layer_ids[]` values instead of assuming dense MoE layer IDs.

## Verification So Far

- `ctest` is not available on PATH in the current shell.
- CMake was found at
  `C:\Users\mdawa\AppData\Local\Android\Sdk\cmake\3.22.1\bin\cmake.exe`.
- Added that CMake directory to the User PATH for future PowerShell windows.
- Rebuilt successfully with:
  `C:\Users\mdawa\AppData\Local\Android\Sdk\cmake\3.22.1\bin\cmake.exe --build build --config Release`.
- `msbuild` is not directly available on PATH, but CMake invokes the Visual
  Studio Build Tools successfully.
- `git status` / `git diff` are not usable from the sandboxed working
  directory; Git reports that this is not a repository.
- `build/Release/test-moe-cache.exe`: passed, 10 tests.
- `build/Release/test-router-config.exe`: passed, 7 tests.
- `build/Release/test-disk-streamer.exe`: passed, 7 tests.
- `build/Release/test-bench.exe`: passed, 1 test.

## Suggested Next Command Sequence

From a normal Windows Developer PowerShell in the repo:

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\test-moe-cache.exe
.\build\Release\test-router-config.exe
.\build\Release\test-disk-streamer.exe
.\build\Release\test-bench.exe
```

If Windows blocks test binaries, follow `README.md` and unblock or add a
Windows Security exclusion for the workspace.

## High-Priority Work

1. Integrate llama.cpp and make `amcoli run -m <model> --prompt "hi"` generate
   real tokens without streaming.
2. Replace hand-estimated GGUF quantized tensor byte sizes with exact sizes from
   llama.cpp/ggml metadata utilities.
3. Use actual transformer layer IDs from `moe_layer_ids[]` consistently instead
   of assuming MoE layers are dense `0..n_moe_layers-1`.
4. Rename or disable the fake VRAM path until it uses real GPU buffers through
   ggml/llama.cpp backend APIs.
5. Implement real asynchronous prefetch after actual router logits are
   available from inference.
6. Keep CLI wording honest: demo/simulated paths should say they are simulated;
   real generation should only be exposed once llama.cpp is wired.

## Notes For Future Agents

- Do not claim the current CLI performs real inference.
- Preserve the existing cache/router tests while refactoring.
- Avoid committing generated build outputs or GGUF model files.
