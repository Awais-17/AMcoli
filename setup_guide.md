# AMcoli — Setup and Installation Guide

This guide covers building, configuring, and troubleshooting the AMcoli engine on your system.

---

## 1. Prerequisites

Before compiling, ensure you have the following installed:
*   **CMake (v3.21+)**: Added to your system PATH.
*   **C++ Compiler**: A compiler supporting C++17:
    *   **Windows**: Visual Studio 2019/2022 Build Tools (MSVC).
    *   **Linux/WSL**: GCC 9+ or Clang 12+.
*   **Curl**: Native `curl.exe` (installed by default on modern Windows 10/11) to handle Hugging Face model downloads.

---

## 2. Quick Global Installation (Recommended)

To run AMcoli globally from any folder in your terminal, select one of the following options:

### Option A: Using PowerShell (Windows)
Run this command in PowerShell to automatically install, add `amcoli` to your `PATH`, and unblock the executable for Smart App Control compatibility:
```powershell
irm https://raw.githubusercontent.com/Awais-17/AMcoli/main/scripts/install.ps1 | iex
```

### Option B: Using NPM (Cross-Platform)
If you have Node.js installed, you can install the package globally from the repository:
```powershell
# Navigate to the npm-wrapper directory
cd npm-wrapper

# Install globally
npm install -g .
```
*(Or install directly from GitHub when published: `npm install -g @github/amcoli`)*

---

## 3. Manual Compilation

Compile the project in Release mode using CMake:

```powershell
# 1. Configure the build system
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build the executable and all test suites
cmake --build build --config Release
```

The compiled binaries will be output to:
*   **Main Runner**: `build/Release/amcoli.exe`
*   **Tests**: `build/Release/test-bench.exe`, `build/Release/test-moe-cache.exe`, etc.

---

## 4. Manual PATH Setup

To launch manually compiled AMcoli from any terminal by typing just `amcoli`, add its directory to your User PATH variable:

### Temporary (Current Session Only)
```powershell
$env:Path += ";C:\Users\mdawa\OneDrive\Desktop\All\AMcoli\build\Release"
```

### Persistent (Across All Sessions)
Run this command once in PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable("PATH", $Env:PATH + ";C:\Users\mdawa\OneDrive\Desktop\All\AMcoli\build\Release", "User")
```
*Note: Restart your terminal window for the change to take effect.*

---

## 5. Troubleshooting: Application Control Policy Blocks

On Windows 11, **Smart App Control** or **Windows Defender Application Control (WDAC)** may block newly compiled binaries, returning this error:
> `Program 'amcoli.exe' failed to run: An Application Control policy has blocked this file`

To resolve this, select one of the following methods:

### Method A: Add a Windows Security Exclusion (Recommended)
You can tell Windows Defender to ignore the workspace folder so it doesn't block your compiled binaries.
1. Open **Windows Start Menu** and type `Windows Security`.
2. Go to **Virus & threat protection** → **Virus & threat protection settings** → **Manage settings**.
3. Scroll down to **Exclusions** and click **Add or remove exclusions**.
4. Click **Add an exclusion** → **Folder**, and select:
   `C:\Users\mdawa\OneDrive\Desktop\All\AMcoli`

### Method B: Unblock the Executable in PowerShell
Sometimes Windows flags files downloaded or generated in user folders. Run:
```powershell
Unblock-File -Path .\build\Release\amcoli.exe
Unblock-File -Path .\build\Release\test-moe-cache.exe
```

### Method C: Smart App Control Settings
If Smart App Control is in "Enforced" mode, it blocks all unsigned executables:
1. Open **Windows Security** → **App & browser control** → **Smart App Control settings**.
2. If it is blocking your custom builds, you can change the state to **Evaluation** or **Off** (note: turning it off requires a system restart and cannot be re-enabled without a Windows reinstall).

