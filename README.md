# AMcoli — Universal MoE Disk-Streaming Inference Engine

**Prototype a disk-streaming runtime for massive GGUF Mixture-of-Experts (MoE)
models on constrained hardware.**

AMcoli treats disk, RAM, and VRAM as a unified memory hierarchy — streaming experts from NVMe on demand, caching hot experts in RAM/VRAM, and prefetching likely next-layer experts to overlap I/O with compute.

AMcoli is designed to integrate with [llama.cpp](https://github.com/ggerganov/llama.cpp)
for model-generic GGUF loading and compute. The current codebase implements the
GGUF inspection, expert indexing, cache, downloader, and benchmark simulation
pieces; real token generation through llama.cpp is still pending.

---

## Key Features

- **Architecture-generic**: Runs Mixtral, Qwen-MoE, DeepSeek-V3, GLM-5.1, and any GGUF MoE model without per-model code changes.
- **Three-tier memory hierarchy**: VRAM (hot) → RAM (warm) → Disk/SSD (cold).
- **Persistent Two-Tier Expert Cache**: Keeps frequently used experts cached close to compute, with LRU/LFU eviction.
- **Speculative Prefetching**: Uses router logits to predict next-layer experts, overlapping disk I/O with computation.
- **Dynamic System Spec Panel**: Auto-detects system specs (CPU cores, RAM size, GPU adapter details via DXGI/Win32, SSD disk space).
- **Intelligent Model Selector**: Highlights compatible models based on your hardware specs.
- **Built-in HF Downloader**: Pulls models directly from Hugging Face with real-time percentage and progress bar terminal animation.

---

## Registered Models

AMcoli contains a built-in downloader registry for the highest-performing open source MoE and Coding families:

| Index | Model Name | Total Params | Active Params | Size (GB) | Focus / Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Qwen1.5-MoE-A2.7B-Chat** | 14.3B | 2.7B | 8.2 GB | Chat MoE |
| 2 | **Qwen3-30B-A3B-Instruct** | 30.5B | 3.3B | 18.2 GB | Chat MoE |
| 3 | **Qwen2-57B-A14B-Instruct** | 57.0B | 14.0B | 34.2 GB | Chat MoE |
| 4 | **DeepSeek-MoE-16B-Chat** | 16.4B | 2.8B | 9.2 GB | Chat MoE |
| 5 | **DeepSeek-Coder-V2-Lite-Instruct** | 16.4B | 2.4B | 10.4 GB | Coding MoE |
| 6 | **DeepSeek-V3** | 671.0B | 37.0B | 23.8 GB | Chat MoE (Extreme) |
| 7 | **JetMoe-8B-Chat** | 8.0B | 2.2B | 4.8 GB | Lightweight MoE |
| 8 | **DBRX-Instruct** | 132.0B | 36.0B | 46.2 GB | Chat MoE |
| 9 | **Nemotron-4-340B-Base** | 340.0B | 68.0B | 96.5 GB | Base MoE |
| 10 | **Nemotron-4-340B-Instruct** | 340.0B | 68.0B | 96.5 GB | Instruct MoE |
| 11 | **Nemotron-4-340B-Reward** | 340.0B | 68.0B | 96.5 GB | Reward MoE |
| 12 | **Grok-1** | 314.0B | 86.0B | 110.3 GB | Large MoE |
| 13 | **Mixtral-8x7B-Instruct-v0.1** | 46.7B | 12.9B | 26.4 GB | Chat MoE |
| 14 | **Mixtral-8x22B-v0.1** | 141.0B | 39.0B | 45.1 GB | Large MoE |
| 15 | **GLM-5.1** | 754.0B | 40.0B | 200.0 GB | Large MoE (Agentic) |
| 16 | **Qwen2.5-Coder-7B-Instruct** | 7.6B | 7.6B | 4.7 GB | Coding (Dense) |
| 17 | **Qwen2.5-Coder-32B-Instruct** | 32.5B | 32.5B | 20.3 GB | Coding (Dense) |

---

## Building and Setup

AMcoli compiles natively on Windows, WSL2, and Linux. See the [Setup Guide](setup_guide.md) for full compile instructions, environment variable mapping, and OS-specific security guidelines.

```bash
# Configure the build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile the binaries
cmake --build build --config Release
```

---

## Usage

Start the interactive visual selector and simulated chat/cache demo:
```bash
# Launch selector CLI
./build/Release/amcoli run
```

Or inspect/run the current simulated path on a local GGUF file:
```bash
# Start the current demo on a downloaded GGUF file
./build/Release/amcoli run --model .models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf
```

Note: `run` does not generate real model tokens yet. It validates/parses the
model, exercises expert cache reads, and streams demo text until llama.cpp
inference integration is added.

---

## Architecture Flow

```
GGUF Loader (generic metadata parse)
        │
Router Config Extractor (n_expert, top_k, shared/routed split)
        │
    ┌───┴───┐
    │Compute│──uses──> Expert Cache (VRAM slots + RAM slots, LRU/LFU)
    │(Fwd)  │                │
    └───┬───┘           miss │ hit
        │                    ▼
        │              Disk Streamer (NVMe, mmap → async I/O)
        │                    │
        └─────prefetch───────┘ (based on router logits)
```

## Developer Context

If you are an AI assistant pair-programming on this codebase, refer to the [Developer Context](context.md) for architectural details, completed milestones, and codebase structure.
