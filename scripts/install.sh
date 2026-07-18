#!/usr/bin/env bash

# AMcoli — Linux & macOS Installer Script
# Installs amcoli globally under ~/.amcoli/bin and registers PATH

set -e

REPO="Awais-17/AMcoli"
INSTALL_DIR="$HOME/.amcoli"
BIN_DIR="$INSTALL_DIR/bin"
BINARY_NAME="amcoli"
DEST_PATH="$BIN_DIR/$BINARY_NAME"

echo -e "\033[1;36m================================================================================\033[0m"
echo -e "                   \033[1;33mAMcoli — Universal MoE Installer (UNIX)\033[0m"
echo -e "\033[1;36m================================================================================\033[0m"

# Create directories
mkdir -p "$BIN_DIR"

# Platform detection
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [[ "$OS" == "darwin" ]]; then
    OS_NAME="macos"
elif [[ "$OS" == "linux" ]]; then
    OS_NAME="linux"
else
    echo -e "\033[1;31mError: Unsupported operating system: $OS\033[0m"
    exit 1
fi

if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    ARCH_NAME="arm64"
else
    ARCH_NAME="x64"
fi

ASSET_NAME="${OS_NAME}-${ARCH_NAME}"
RELEASE_URL="https://github.com/${REPO}/releases/latest/download/${ASSET_NAME}"

echo -e "\033[1;32m[SYSTEM]: Detected Platform: ${OS_NAME} (${ARCH_NAME})\033[0m"

DOWNLOAD_SUCCESS=false

# Attempt download
if command -v curl >/dev/null 2>&1; then
    echo -e "[STATUS]: Downloading precompiled binary via curl..."
    if curl -fsSL -H "User-Agent: AMcoli-Bash-Installer" -o "$DEST_PATH" "$RELEASE_URL"; then
        DOWNLOAD_SUCCESS=true
    fi
elif command -v wget >/dev/null 2>&1; then
    echo -e "[STATUS]: Downloading precompiled binary via wget..."
    if wget -q -O "$DEST_PATH" "$RELEASE_URL"; then
        DOWNLOAD_SUCCESS=true
    fi
fi

if [ "$DOWNLOAD_SUCCESS" = true ]; then
    chmod +x "$DEST_PATH"
    echo -e "\033[1;32m[SUCCESS]: Precompiled binary installed to $DEST_PATH\033[0m"
else
    echo -e "\033[1;33m[WARNING]: Precompiled binary not found or download failed. Falling back to compiling from source...\033[0m"
    
    # Check dependencies
    if ! command -v cmake >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        echo -e "\033[1;31mError: Compilation requires 'git' and 'cmake' to be installed on your system.\033[0m"
        exit 1
    fi
    
    BUILD_TEMP=$(mktemp -d)
    echo -e "[STATUS]: Cloning repository..."
    git clone --depth 1 https://github.com/${REPO}.git "$BUILD_TEMP"
    
    echo -e "[STATUS]: Configuring build with CMake..."
    cmake -B "$BUILD_TEMP/build" -S "$BUILD_TEMP" -DCMAKE_BUILD_TYPE=Release
    
    echo -e "[STATUS]: Compiling binaries..."
    cmake --build "$BUILD_TEMP/build" --config Release
    
    # Copy build
    cp "$BUILD_TEMP/build/amcoli" "$DEST_PATH"
    chmod +x "$DEST_PATH"
    
    # Clean up temp dir
    rm -rf "$BUILD_TEMP"
    echo -e "\033[1;32m[SUCCESS]: Compiled and installed to $DEST_PATH\033[0m"
fi

# Add PATH setup guidance
SHELL_PROFILE=""
case "$SHELL" in
    */bash)
        if [ -f "$HOME/.bashrc" ]; then SHELL_PROFILE="$HOME/.bashrc"; else SHELL_PROFILE="$HOME/.bash_profile"; fi
        ;;
    */zsh)
        SHELL_PROFILE="$HOME/.zshrc"
        ;;
    *)
        SHELL_PROFILE="$HOME/.profile"
        ;;
esac

PATH_CMD="export PATH=\"\$HOME/.amcoli/bin:\$PATH\""

if ! grep -q ".amcoli/bin" "$SHELL_PROFILE" 2>/dev/null; then
    echo -e "\n[STATUS]: Registering PATH in $SHELL_PROFILE..."
    echo -e "\n# AMcoli PATH setup\n$PATH_CMD" >> "$SHELL_PROFILE"
    echo -e "\033[1;32m[SUCCESS]: Added AMcoli to PATH.\033[0m"
else
    echo -e "\n[STATUS]: PATH already registered in $SHELL_PROFILE."
fi

echo -e "\n\033[1;36m================================================================================\033[0m"
echo -e "                   \033[1;32mInstallation Completed Successfully!\033[0m"
echo -e "\033[1;36m================================================================================\033[0m"
echo -e "To start using AMcoli, please run:"
echo -e "  \033[1;33msource $SHELL_PROFILE\033[0m"
echo -e "  \033[1;33mamcoli run\033[0m"
echo -e "\033[1;36m================================================================================\033[0m\n"
