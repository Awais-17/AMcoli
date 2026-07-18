# AMcoli — Developer Context

This document is designed for AI coding assistants (like Claude Code, Codex, or Antigravity) to instantly understand the state, architecture, achievements, and roadmap of the AMcoli project.

---

## 1. Project Motive
Mixture-of-Experts (MoE) models are highly efficient during inference because only a fraction of parameters (e.g., 2.7B out of 14B, or 37B out of 671B) are activated per token. However, standard runtimes require loading the *entire* model weight file (tens or hundreds of gigabytes) into RAM or VRAM, making them impossible to run on consumer hardware (such as laptops with 16GB RAM and 4GB GPU VRAM).

**AMcoli** solves this by treating system storage (NVMe SSD), RAM, and VRAM as a single unified memory hierarchy. It loads only the non-MoE weights (attention, embeddings) into RAM/VRAM, and streams expert layers on-demand from SSD during the forward pass, using a two-tier cache to keep active experts resident in memory and overlapping disk read operations with compute via prefetching.

---

## 2. Core Architecture
The codebase is divided into two primary parts:
1.  **AMcoli Core Library (`src/`)**:
    *   [llama-disk-streamer.cpp](src/llama-disk-streamer.cpp): Manages SSD memory-mapping and asynchronous expert layer fetching.
    *   [llama-moe-cache.cpp](src/llama-moe-cache.cpp): Implements the two-tier (VRAM + RAM) cache containing slot insertion, LRU/LFU eviction, and cache hits.
    *   [amcoli-downloader.cpp](src/amcoli-downloader.cpp): Handles Hugging Face GGUF registry lookups, downloads via Curl, and terminal percentage bars.
    *   [amcoli-sys-info.cpp](src/amcoli-sys-info.cpp): System specs detection (Win32 DXGI for GPU names, CPU core parsing, RAM status).
2.  **CLI Interface (`tools/run/`)**:
    *   [main.cpp](tools/run/main.cpp): Program entry point. Houses the console UTF-8 setup, loop menu selector, recommendation guide, and model loading/harness wrapper.

---

## 3. Achievements & Completed Milestones
*   **Dynamic Hardware Auto-Detection**: Auto-detects available CPU cores, RAM limits, active drive storage, and queries active Windows graphics cards via DXGI (specifically recognizing cards like the *RTX 3050 Laptop GPU*).
*   **Hardware-Aware Recommendation Engine**: Evaluates available RAM and GPU capacity at startup to suggest the optimal model for the user's specific specs (recommended models are flagged green in the registry table).
*   **Robust Input Selector Loop**: Fully insulated console loop that handles invalid commands, typos, and quit directives without crashing the terminal.
*   **Extended Model Registry**: Added full suites for the Qwen MoE family, DeepSeek MoE family, GLM-5.1 MoE, and official Qwen Coder models (17 entries in total) with verified download links.
*   **Carriage Return (`\r`) Progress Meter**: Implemented animated progress percentage bars inside curl subprocesses for clean CLI visual feedback.
*   **Windows Terminal Compatibility**: Solved UTF-8 box characters corruption by forcing console code page output to UTF-8 (`SetConsoleOutputCP(65001)`) at startup.

---

## 4. Current Working Status & Roadmaps
*   **In-Progress**: Connecting the router config extractor to parse dense vs. routed layers dynamically from custom GGUF models.
*   **Planned**: Speculative prefetching using top-$k$ router logit history to fetch next-layer experts asynchronously before the active layer compute finishes.
