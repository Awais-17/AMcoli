```text
================================================================================

 ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ 
█       █  █▄█  █       █       █  █ █  █       █
█   ▄   █   █   █       █   ▄   █  █▄█  █   ▄   █
█  █▄█  █       █     ▄▄█  █ █  █       █  █ █  █
█       █       █    █  █  █▄█  █       █  █▄█  █
█   ▄   █ ██▄██ █    █▄▄█       █   ▄   █       █
█▄▄█ █▄▄█▄█   █▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█

================================================================================
```

# AMcoli — Universal MoE Disk-Streaming Inference Engine

**Prototype a disk-streaming runtime for massive GGUF Mixture-of-Experts (MoE)
models on constrained hardware.**

AMcoli treats disk, RAM, and VRAM as a unified memory hierarchy — streaming experts from NVMe on demand, caching hot experts in RAM/VRAM, and prefetching likely next-layer experts to overlap I/O with compute.

AMcoli is integrated with [llama.cpp](https://github.com/ggerganov/llama.cpp) for model-generic GGUF loading, tokenization, and local generation. The codebase implements the GGUF inspection, expert indexing, active memory cache management, NVMe streamer, and interactive token-decoding loops.

---

## About AMcoli: The Simple Explanation 💡
<img width="1282" height="751" alt="Screenshot 2026-07-18 130438" src="https://github.com/user-attachments/assets/cd7d38c7-8d66-485b-a166-d1aa0eef062e" />


Imagine you want to run a massive, super-smart AI on your average laptop. Usually, your computer would crash because the AI model is simply too big to fit inside your computer's active memory (RAM/VRAM). 

Here is how **AMcoli** solves this problem using a simple analogy:

### 1. The Analogy: A Team of Specialized Chefs
Think of a giant AI model as a restaurant menu with 30 specialized chefs (called **"Experts"** in a **Mixture-of-Experts / MoE** model). One chef is great at coding C++, another at translation, another at math, etc.
*   **The Problem**: Your laptop is like a **tiny kitchen counter** (VRAM/RAM). There is only enough physical space for 2 or 3 chefs to stand and work at a time. If all 30 chefs try to squeeze into the kitchen at once, the kitchen collapses (your computer runs out of memory and crashes).
*   **The MoE Trick**: When you ask the AI a question, you actually only need **2 or 3 specific chefs** to answer it. The other 27 chefs are just standing there doing nothing.

### 2. How AMcoli Makes It Work
AMcoli acts as the smart head chef managing this tiny kitchen:

*   **Disk Streaming (The Hotel)**: AMcoli keeps all 30 chefs waiting in a nearby hotel (your laptop's hard drive / SSD).
*   **On-Demand Calling**: When you type a prompt, AMcoli instantly figures out which 2 chefs are needed, calls them from the hotel, and places them at the kitchen counter (VRAM/RAM) to do the work. Once they finish, they go back to the hotel.
*   **Caching (Keeping Favorites)**: If a chef is constantly needed (like the coding chef), AMcoli lets them stay in the kitchen permanently instead of sending them back and forth.
*   **Prefetching (Looking Ahead)**: While Chef A is cooking, AMcoli looks at what they are making and says, *"Aha, they will need Chef B in 5 seconds."* AMcoli calls Chef B from the hotel *before* they are actually needed, so Chef B walks in right on time.

By swapping these specialized "experts" in and out of your laptop's memory from the hard drive, **AMcoli lets you run massive, world-class AI models on a standard laptop** that would normally require a expensive $10,000 server.

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
| 16 | **GLM-5.2-Colibri** | 744.0B | 40.0B | 238.0 GB | Large MoE (Agentic) |
| 17 | **Kimi-K2.6** | 1000.0B | 68.0B | 350.0 GB | Large MoE (Agentic) |
| 18 | **Kimi-K3** | 2800.0B | 120.0B | 980.0 GB | Large MoE (Extreme) |
| 19 | **Kimi-Dev-72B-Coder** | 72.5B | 72.5B | 42.5 GB | Coding (Dense) |
| 20 | **Kimi-Coder-135M** | 0.13B | 0.13B | 0.1 GB | Coding (Dense) |
| 21 | **Qwen2.5-Coder-7B-Instruct** | 7.6B | 7.6B | 4.7 GB | Coding (Dense) |
| 22 | **Qwen2.5-Coder-32B-Instruct** | 32.5B | 32.5B | 20.3 GB | Coding (Dense) |
| 23 | **Meta-Llama-3.1-8B-Instruct** | 8.0B | 8.0B | 4.9 GB | General (Dense) |
| 24 | **Meta-Llama-3.1-70B-Instruct** | 70.6B | 70.6B | 43.0 GB | General (Dense) |
| 25 | **Llama-3.2-1B-Instruct** | 1.2B | 1.2B | 1.2 GB | General (Dense) |
| 26 | **Llama-3.2-3B-Instruct** | 3.2B | 3.2B | 2.0 GB | General (Dense) |
| 27 | **Gemma-2-2b-it** | 2.6B | 2.6B | 1.7 GB | General (Dense) |
| 28 | **Gemma-2-9b-it** | 9.2B | 9.2B | 5.7 GB | General (Dense) |
| 29 | **Gemma-2-27b-it** | 27.2B | 27.2B | 17.4 GB | General (Dense) |
| 30 | **Qwen2.5-7B-Instruct** | 7.6B | 7.6B | 4.7 GB | General (Dense) |
| 31 | **Qwen2.5-72B-Instruct** | 72.5B | 72.5B | 46.5 GB | General (Dense) |
| 32 | **Command-R-Plus** | 104.0B | 104.0B | 62.0 GB | Large MoE |
| 33 | **Phi-3-mini-128k-instruct** | 3.8B | 3.8B | 2.4 GB | General (Dense) |
| 34 | **Phi-4-Instruct** | 14.7B | 14.7B | 8.5 GB | General (Dense) |

---

## Installation and Build Guide

This section covers installing, building, configuring, and troubleshooting the AMcoli engine on your system.

### 1. Prerequisites
Before installing or compiling, ensure you have the following ready:
*   **CMake (v3.21+)**: Added to your system PATH (required for compilation).
*   **C++ Compiler**: A compiler supporting C++17:
    *   **Windows**: Visual Studio 2019/2022 Build Tools (MSVC).
    *   **Linux/WSL**: GCC 9+ or Clang 12+.
*   **Curl**: Native `curl.exe` (installed by default on Windows 10/11) to handle Hugging Face model downloads.

### 2. Quick Global Installation (Recommended)
To run AMcoli globally from any folder in your terminal, select the appropriate option for your operating system:

#### Option A: Windows (PowerShell)
Run this command in PowerShell to automatically install, add `amcoli` to your `PATH`, and unblock the executable for Smart App Control compatibility:
```powershell
irm https://raw.githubusercontent.com/Awais-17/AMcoli/main/scripts/install.ps1 | iex
```

#### Option B: Linux & macOS (Bash)
Run this command in your terminal to automatically compile/install, register the `amcoli` binary inside `~/.amcoli/bin`, and add it to your shell profile (`.bashrc` or `.zshrc`):
```bash
curl -fsSL https://raw.githubusercontent.com/Awais-17/AMcoli/main/scripts/install.sh | bash
```

#### Option C: Node.js / NPM (Cross-Platform)
If you have Node.js installed, you can compile and install the package globally directly from the source directory:
```bash
# Navigate to the npm-wrapper directory
cd npm-wrapper

# Install globally
npm install -g .
```
*(Or install directly from GitHub: `npm install -g github:Awais-17/AMcoli#main`)*


### 3. Manual Compilation
If you prefer to compile manually in Release mode using CMake:
```powershell
# 1. Configure the build system
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build the executable and all test suites
cmake --build build --config Release
```
The compiled binaries will be output to:
*   **Main Runner**: `build/Release/amcoli.exe`
*   **Tests**: `build/Release/test-bench.exe`, `build/Release/test-moe-cache.exe`, etc.

### 4. Manual PATH Setup (Optional)
To launch manually compiled AMcoli from any terminal by typing just `amcoli`, add its directory to your User PATH variable:

#### Temporary (Current Session Only)
Run this command in PowerShell from the repository root:
```powershell
$env:Path += ";$(Get-Item -Path '.\build\Release').FullName"
```

#### Persistent (Across All Sessions)
Run this command in PowerShell from the repository root to automatically resolve and persistently register the directory:
```powershell
[System.Environment]::SetEnvironmentVariable("PATH", $Env:PATH + ";$(Get-Item -Path '.\build\Release').FullName", "User")
```
*Note: Restart your terminal window for the change to take effect.*

### 5. Troubleshooting: Application Control Policy Blocks
On Windows 11, **Smart App Control** or **Windows Defender Application Control (WDAC)** may block newly compiled binaries, returning this error:
> `Program 'amcoli.exe' failed to run: An Application Control policy has blocked this file`

To resolve this, select one of the following methods:

#### Method A: Add a Windows Security Exclusion (Recommended)
You can tell Windows Defender to ignore the workspace folder so it doesn't block your compiled binaries.
1. Open **Windows Start Menu** and type `Windows Security`.
2. Go to **Virus & threat protection** → **Virus & threat protection settings** → **Manage settings**.
3. Scroll down to **Exclusions** and click **Add or remove exclusions**.
4. Click **Add an exclusion** → **Folder**, and select your local AMcoli directory (e.g. `C:\path\to\AMcoli`).

#### Method B: Unblock the Executable in PowerShell
Sometimes Windows flags files downloaded or generated in user folders. Run:
```powershell
Unblock-File -Path .\build\Release\amcoli.exe
Unblock-File -Path .\build\Release\test-moe-cache.exe
```

#### Method C: Smart App Control Settings
If Smart App Control is in "Enforced" mode, it blocks all unsigned executables:
1. Open **Windows Security** → **App & browser control** → **Smart App Control settings**.
2. If it is blocking your custom builds, you can change the state to **Evaluation** or **Off** (note: turning it off requires a system restart and cannot be re-enabled without a Windows reinstall).

---

## Usage

### 1. Agentic Workflow CLI (Claude Code style)
You can launch an autonomous, agentic coding assistant that can run local commands, view files, and list directories with your explicit approval:
```bash
# Start the agent using default Ollama server and Qwen-Coder model
amcoli agent

# Or configure a custom API endpoint and model
amcoli agent --api-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b
```

### 2. Native Engine CLI
Start the C++ interactive visual selector and simulated chat/cache demo:
```bash
# Launch selector CLI
amcoli run
```

Or inspect/run the current simulated path on a local GGUF file:
```bash
# Start the current demo on a downloaded GGUF file
amcoli run --model .models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf
```

*Note: `run` performs real model token generation using the linked `llama.cpp` inference engine, while dynamically updating the active RAM/VRAM expert cache statistics.*

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
