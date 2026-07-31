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

AMcoli treats disk, RAM, and VRAM as a unified memory hierarchy — streaming experts from NVMe on demand, caching hot experts in RAM/VRAM, and prefetching likely next-layer experts to overlap I/O with computation.

AMcoli is integrated with [llama.cpp](https://github.com/ggerganov/llama.cpp) for model-generic GGUF loading, tokenization, and local generation. The codebase implements the GGUF inspection, expert indexing, active memory cache management, NVMe streamer, and interactive token-decoding loops.

**Current version: 0.1.1** — see the [Release](https://github.com/Awais-17/AMcoli/releases) page.

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

- **Architecture-generic**: Runs Mixtral, Qwen-MoE, DeepSeek, GLM, Kimi, Phi-MoE, OLMoE, Granite, Jamba, DBRX, and any GGUF MoE model without per-model code changes.
- **Three-tier memory hierarchy**: VRAM (hot) → RAM (warm) → Disk/SSD (cold).
- **Persistent Two-Tier Expert Cache**: Keeps frequently used experts cached close to compute, with LRU/LFU eviction.
- **Speculative Prefetching**: Uses router logits to predict next-layer experts, overlapping disk I/O with computation.
- **Dynamic System Spec Panel**: Auto-detects system specs (CPU cores, RAM size, GPU adapter details via DXGI/Win32, SSD disk space).
- **Intelligent Model Selector**: Highlights compatible models based on your hardware specs.
- **Built-in HF Downloader**: Pulls models directly from Hugging Face with a real-time progress bar; **resumable** (`-C -`) downloads survive interruptions.
- **36-Model Registry + Health Check**: `amcoli list` prints the full catalog, `amcoli check` verifies every registry URL is reachable, and `amcoli info <alias>` shows details for one model.
- **Automatic Update Notifications**: Checks once per day (in the background) whether a newer AMcoli release exists.
- **Agentic Coding Assistant** (`amcoli agent`): A local AI assistant (via the npm wrapper) with real filesystem + shell tools, per-call approval, session history, and optional streaming.

---

## Registered Models

AMcoli contains a built-in downloader registry of 36 models across the highest-performing open-source MoE and Coding/Dense families. Each row shows the `pull` alias, the quantized GGUF used, total/active parameter counts, and download size.

Run `amcoli list` to print this table in the terminal, `amcoli info <alias>` for one model's details, and `amcoli check` to verify every URL is reachable.

| Alias | Model (Quant) | Total | Active | Size (GB) | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen-3b` | Qwen1.5-MoE-A2.7B-Chat (Q4_K_M) | 14.3B | 2.7B | 8.84 | Chat MoE |
| `qwen-14b` | Qwen3-30B-A3B-Instruct (Q4_K_M) | 30.5B | 3.3B | 17.28 | Chat MoE |
| `qwen-57b` | Qwen2-57B-A14B-Instruct (Q4_K_M) | 57.0B | 14.0B | 32.46 | Chat MoE |
| `deepseek-moe-16b` | DeepSeek-MoE-16B-Chat (Q4_K_M) | 16.4B | 2.8B | 9.20 | Chat MoE |
| `deepseek-lite` | DeepSeek-Coder-V2-Lite-Instruct (Q4_K_M) | 16.4B | 2.4B | 10.40 | Coding MoE |
| `dbrx` | DBRX-Instruct (IQ2_XXS) | 132.0B | 36.0B | 32.24 | Chat MoE |
| `mixtral` | Mixtral-8x7B-Instruct-v0.1 (Q4_K_M) | 46.7B | 12.9B | 26.40 | Chat MoE |
| `mixtral-8x22b` | Mixtral-8x22B-v0.1 (IQ2_XXS) | 141.0B | 39.0B | 35.28 | Large MoE |
| `qwen-coder-30b` | Qwen3-Coder-30B-A3B-Instruct (Q4_K_M) | 30.5B | 3.3B | 17.28 | Coding MoE |
| `phi-3.5-moe` | Phi-3.5-MoE-instruct (Q4_K_M) | 42.0B | 6.6B | 23.61 | Chat MoE |
| `phi-4-moe` | Phi-4-MoE-2x14B-Instruct (Q4_K_M) | 40.0B | 5.6B | 23.06 | Chat MoE |
| `olmoe-7b` | OLMoE-1B-7B-0924-Instruct (Q4_K_M) | 7.0B | 1.3B | 3.92 | Lightweight MoE |
| `granite-3b-a800m` | Granite-3.0-3B-A800M-Instruct (Q4_K_M) | 3.3B | 0.8B | 1.92 | Lightweight MoE |
| `deepseek-v2-lite` | DeepSeek-V2-Lite-Chat (Q4_K_M) | 16.4B | 2.4B | 9.65 | Chat MoE |
| `jamba-mini-1.5` | AI21-Jamba-1.5-Mini (Q4_K_M) | 52.0B | 12.0B | 29.02 | Chat MoE |
| `deepseek-v3.1` | DeepSeek-V3.1 (UD-TQ1_0) | 671.0B | 37.0B | 158.80 | Large MoE (Extreme) |
| `glm-5.1` | GLM-5.1 (IQ2_XXS) | 754.0B | 40.0B | 200.00 | Large MoE (Agentic) |
| `glm-5.2` | GLM-5.2-Colibri (UD-IQ2_XXS) | 744.0B | 40.0B | 238.00 | Large MoE (Agentic) |
| `kimi-k2.6` | Kimi-K2.6 (UD-Q2_K_XL) | 1000.0B | 68.0B | 350.00 | Large MoE (Agentic) |
| `kimi-k3` | Kimi-K3 (UD-Q2_K_XL) | 2800.0B | 120.0B | 980.00 | Large MoE (Extreme) |
| `kimi-coder-72b` | Kimi-Dev-72B-Coder (IQ4_NL) | 72.5B | 72.5B | 38.48 | Coding (Dense) |
| `kimi-coder-135m` | Kimi-Coder-135M (Q4_K_M) | 0.14B | 0.14B | 0.10 | Coding (Dense) |
| `qwen-coder-7b` | Qwen2.5-Coder-7B-Instruct (Q4_K_M) | 7.6B | 7.6B | 4.70 | Coding (Dense) |
| `qwen-coder-32b` | Qwen2.5-Coder-32B-Instruct (Q4_K_M) | 32.5B | 32.5B | 20.30 | Coding (Dense) |
| `llama-3.1-8b` | Meta-Llama-3.1-8B-Instruct (Q4_K_M) | 8.0B | 8.0B | 4.90 | General (Dense) |
| `llama-3.1-70b` | Meta-Llama-3.1-70B-Instruct (Q4_K_M) | 70.6B | 70.6B | 43.00 | General (Dense) |
| `llama-3.2-1b` | Llama-3.2-1B-Instruct (Q4_K_M) | 1.2B | 1.2B | 1.20 | General (Dense) |
| `llama-3.2-3b` | Llama-3.2-3B-Instruct (Q4_K_M) | 3.2B | 3.2B | 2.00 | General (Dense) |
| `gemma-2-2b` | Gemma-2-2b-it (Q4_K_M) | 2.6B | 2.6B | 1.70 | General (Dense) |
| `gemma-2-9b` | Gemma-2-9b-it (Q4_K_M) | 9.2B | 9.2B | 5.70 | General (Dense) |
| `gemma-2-27b` | Gemma-2-27b-it (Q4_K_M) | 27.2B | 27.2B | 17.40 | General (Dense) |
| `qwen-2.5-7b` | Qwen2.5-7B-Instruct (Q4_K_M) | 7.6B | 7.6B | 4.36 | General (Dense) |
| `qwen-2.5-72b` | Qwen2.5-72B-Instruct (Q4_K_M) | 72.5B | 72.5B | 44.16 | General (Dense) |
| `command-r-plus` | Command-R-Plus (IQ3_M) | 104.0B | 104.0B | 44.41 | General (Dense) |
| `phi-3-mini` | Phi-3-mini-128k-instruct (Q4_K_M) | 3.8B | 3.8B | 2.23 | General (Dense) |
| `phi-4` | Phi-4-Instruct (Q4_K_M) | 14.7B | 14.7B | 8.50 | General (Dense) |

*Multi-shard models (e.g. GLM-5.1, Kimi-K3) are registered by their first shard file — `amcoli pull <alias>` fetches that single file.*

---

## Installation and Build Guide
### 0.1 official website:
  Download From : https://amcoli-exe.vercel.app/

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
You can launch an autonomous, agentic coding assistant that reads, searches, edits files and runs local commands — with your explicit approval:
```bash
# Start the agent using the default local Ollama server
amcoli agent

# Configure a custom API endpoint, model, and API key
amcoli agent --api-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b --api-key sk-...
```

**Flags:**
```bash
amcoli agent [options]
  -u, --api-url <url>   OpenAI-compatible API endpoint (default: http://127.0.0.1:11434/v1)
  -m, --model <name>    Model name (default: qwen-coder-32b)
  -k, --api-key <key>   API key for the endpoint
  -y, --yes             Auto-approve file modifications and command execution
      --auto            Auto-approve read-only tools, prompt for the rest
      --stream          Stream model output as it is generated
```

These can also be set via the `AMCOLI_API_URL`, `AMCOLI_MODEL`, and `AMCOLI_API_KEY` environment variables.

**Tools** the agent can call on your behalf:
| Tool | Purpose |
| :--- | :--- |
| `list_dir` | List a directory's contents |
| `read_file` | Read a text file (line ranges; 5 MB guard) |
| `search_files` | Substring/regex search across a directory tree |
| `get_system_info` | CPU / RAM / disk / OS facts |
| `write_file` | Create or overwrite a file |
| `edit_file` | Find-and-replace edit (`all` for every occurrence) |
| `run_command` | Run a shell command (compile, test, git, …) |
| `run_amcoli` | Run the local AMcoli binary (`list`, `check`, `info`, `bench`, …) |

**Slash commands** inside the session: `/help`, `/clear`, `/reset`, `/model [name]`, `/api [url]`, `/quit` (or `q`).

Session history persists to `~/.amcoli/agent-history.jsonl` so conversations resume across sessions; `/reset` clears it.

### 2. Native Engine CLI
Start the C++ interactive visual selector and live chat/cache execution:
```bash
# Launch selector CLI (pick a model → download → chat)
amcoli run
```

Or run the live inference path on a local GGUF file:
```bash
# Run local token generation on a downloaded GGUF file
amcoli run --model .models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf
```

*Note: `run` performs real model token generation using the linked `llama.cpp` inference engine, while dynamically updating the active RAM/VRAM expert cache statistics.*

### 3. Command Reference

| Command | Purpose |
| :--- | :--- |
| `amcoli` | Interactive model selector → download → chat |
| `amcoli run` | Chat loop (real llama.cpp token decoding + live cache stats) |
| `amcoli agent` | Agentic coding assistant (real filesystem + shell tools; see §1) |
| `amcoli pull <alias>` | Download a model from the registry (resumable; `--dry-run` previews it) |
| `amcoli list` | Print the full 36-model registry table |
| `amcoli check` | Verify all 36 registry URLs are reachable |
| `amcoli info <alias>` | Show download URL + details for one model |
| `amcoli recommend` | Print hardware-based model recommendation |
| `amcoli bench` | Run the Zipf benchmark over the expert cache |
| `amcoli version` / `--version` | Print the version string |
| `amcoli serve` / `convert` | Placeholders (Phase 5, not yet implemented) |

Inside the chat loop and selector, type `/help` to list slash commands (`/exit`, `/stats`, `/memory`, `/clear`, `/version`, `/recommend`, …).

AMcoli automatically checks (once per day, in the background) whether a newer release exists and prints a notice when one is available.

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
