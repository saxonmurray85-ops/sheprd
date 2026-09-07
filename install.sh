#!/usr/bin/env bash
# ==============================================================================
# Fold: Universal Linux Installer & Setup Script
# Installs llama.cpp Vulkan runtime, Fold environment, and registers the CLI.
# Supports Ubuntu, Debian, Fedora, Arch, openSUSE, Alpine, and other distros.
# ==============================================================================

set -euo pipefail

# ANSI Pastel Styling (Sky Blue, Sage Green, Slate Grey, Cream)
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_BLUE="\033[38;5;117m"
C_GREEN="\033[38;5;151m"
C_BRIGHT_BLUE="\033[38;5;153m"
C_DIM="\033[2m"
C_SLATE="\033[38;5;248m"
C_AMBER="\033[38;5;222m"
C_RED="\033[38;5;217m"

echo -e "${C_BLUE}"
cat << 'BANNER'
  _____ ___  _     ____  
 |  ___/ _ \| |   |  _ \ 
 | |_ | | | | |   | | | |
 |  _|| |_| | |___| |_| |
 |_|   \___/|_____|____/ 
BANNER
echo -e "${C_RESET}"
echo -e "${C_GREEN}${C_BOLD}Installing Fold // llama.cpp Agent Orchestrator...${C_RESET}\n"

# Helper: download file with curl or wget
download_file() {
    local url="$1"
    local dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -sSL -o "${dest}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${dest}" "${url}"
    else
        return 1
    fi
}

# 1. Dependency and Platform Checks
echo -e "${C_DIM}[1/5] Checking prerequisites...${C_RESET}"

# Verify Python
command -v python3 >/dev/null 2>&1 || {
    echo -e "${C_RED}Error: python3 is required.${C_RESET}"
    exit 1
}

# Verify Python version >= 3.10
python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo -e "${C_RED}Error: Python 3.10 or higher is required.${C_RESET}"
    exit 1
}

# Verify download tool
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo -e "${C_RED}Error: Neither 'curl' nor 'wget' was found. Please install one of them.${C_RESET}"
    exit 1
fi

# Verify extraction tool
command -v tar >/dev/null 2>&1 || {
    echo -e "${C_RED}Error: 'tar' is required to extract binary bundles.${C_RESET}"
    exit 1
}

# Architecture check
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64|amd64)
        LLAMA_ARCH_SUPPORTED=true
        ;;
    *)
        LLAMA_ARCH_SUPPORTED=false
        echo -e "${C_AMBER}Notice: Detected system architecture '${ARCH}'. Prebuilt Vulkan runtime is built for x86_64.${C_RESET}"
        ;;
esac

# 2. Directory Layout
echo -e "${C_DIM}[2/5] Creating directory layout...${C_RESET}"
FOLD_HOME="${HOME}/.local/share/fold"
FOLD_BIN="${FOLD_HOME}/bin"
FOLD_MODELS="${FOLD_HOME}/models"
FOLD_LOGS="${FOLD_HOME}/logs"
FOLD_LOCKS="${FOLD_HOME}/locks"
FOLD_VENV="${FOLD_HOME}/venv"
LOCAL_BIN="${HOME}/.local/bin"

mkdir -p "${FOLD_BIN}" "${FOLD_MODELS}" "${FOLD_LOGS}" "${FOLD_LOCKS}" "${LOCAL_BIN}" "${HOME}/.config/fold"
chmod 700 "${FOLD_HOME}" "${HOME}/.config/fold" 2>/dev/null || true

# Check for legacy Sheprd data migration
LEGACY_SHEPRD_HOME="${HOME}/.local/share/sheprd"
if [ -d "${LEGACY_SHEPRD_HOME}/models" ] && [ ! -d "${FOLD_MODELS}/qwen2.5-0.5b-instruct-q4_k_m.gguf" ]; then
    echo -e "${C_SLATE}  Found legacy models in ${LEGACY_SHEPRD_HOME}/models; preserving access.${C_RESET}"
fi

# 3. Check / Download llama.cpp Runtime
echo -e "${C_DIM}[3/5] Checking llama.cpp binary bundle...${C_RESET}"
if [ -f "${FOLD_BIN}/llama-server" ]; then
    echo -e "${C_GREEN}  llama-server runtime is already present at ${FOLD_BIN}.${C_RESET}"
elif [ -f "${LEGACY_SHEPRD_HOME}/bin/llama-server" ]; then
    echo -e "${C_GREEN}  Using existing llama-server runtime from ${LEGACY_SHEPRD_HOME}/bin.${C_RESET}"
    cp -r "${LEGACY_SHEPRD_HOME}/bin/"* "${FOLD_BIN}/" 2>/dev/null || true
elif [ "${LLAMA_ARCH_SUPPORTED}" = true ]; then
    echo -e "${C_BLUE}  Downloading prebuilt llama.cpp (Vulkan/x86_64 b10827)...${C_RESET}"
    LLAMA_RELEASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/b10827/llama-b10827-bin-ubuntu-vulkan-x64.tar.gz"
    EXPECTED_SHA256="9005df90e98f94bbf20b328104e4481a1a09fbc47c3039617923ffb1287fdcc4"
    TEMP_TAR="/tmp/fold-llama-$$.tar.gz"

    if download_file "${LLAMA_RELEASE_URL}" "${TEMP_TAR}"; then
        ACTUAL_SHA256=""
        if command -v sha256sum >/dev/null 2>&1; then
            ACTUAL_SHA256="$(sha256sum "${TEMP_TAR}" | awk '{print $1}')"
        elif command -v shasum >/dev/null 2>&1; then
            ACTUAL_SHA256="$(shasum -a 256 "${TEMP_TAR}" | awk '{print $1}')"
        fi

        if [ -n "${ACTUAL_SHA256}" ] && [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
            echo -e "${C_RED}  Error: SHA-256 checksum mismatch for llama.cpp binary bundle!${C_RESET}"
            rm -f "${TEMP_TAR}"
            exit 1
        fi

        tar -xzf "${TEMP_TAR}" -C "${FOLD_BIN}" --strip-components=1
        rm -f "${TEMP_TAR}"
        echo -e "${C_GREEN}  llama-server runtime verified and installed to ${FOLD_BIN}.${C_RESET}"
    else
        echo -e "${C_AMBER}  Notice: Could not download prebuilt bundle. Fold will look for llama-server on PATH or in ${FOLD_BIN}.${C_RESET}"
    fi
else
    echo -e "${C_AMBER}  Notice: For architecture '${ARCH}', place a compatible llama-server binary in ${FOLD_BIN}/ or install it to your system PATH.${C_RESET}"
fi

# 4. Install Fold Package & Python Dependencies
echo -e "${C_DIM}[4/5] Installing Fold environment and dependencies...${C_RESET}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_METHOD=""
PYTHON_TARGET=""

# Strategy A: Dedicated Virtual Environment (PEP 668 compliant across Ubuntu 24.04, Debian 12+, Arch, Fedora, etc.)
if python3 -m venv "${FOLD_VENV}" 2>/dev/null; then
    echo -e "${C_DIM}  Creating isolated virtual environment at ${FOLD_VENV}...${C_RESET}"
    if "${FOLD_VENV}/bin/pip" install --quiet -e "${SCRIPT_DIR}" 2>/dev/null; then
        INSTALL_METHOD="venv"
        PYTHON_TARGET="${FOLD_VENV}/bin/python"
        echo -e "${C_GREEN}  ✓ Installed Fold and dependencies inside isolated venv.${C_RESET}"
    fi
fi

# Strategy B: pip with --break-system-packages (if venv creation was unavailable)
if [ -z "${INSTALL_METHOD}" ]; then
    if python3 -m pip install --user --break-system-packages -e "${SCRIPT_DIR}" 2>/dev/null; then
        INSTALL_METHOD="pip_break"
        PYTHON_TARGET="$(command -v python3)"
        echo -e "${C_GREEN}  ✓ Installed Fold in user mode (--break-system-packages).${C_RESET}"
    fi
fi

# Strategy C: Standard pip install --user (for older distros without PEP 668)
if [ -z "${INSTALL_METHOD}" ]; then
    if python3 -m pip install --user -e "${SCRIPT_DIR}" 2>/dev/null; then
        INSTALL_METHOD="pip_user"
        PYTHON_TARGET="$(command -v python3)"
        echo -e "${C_GREEN}  ✓ Installed Fold in user mode via pip.${C_RESET}"
    fi
fi

# Strategy D: Check if system python already has aiohttp
if [ -z "${INSTALL_METHOD}" ]; then
    if python3 -c "import aiohttp" 2>/dev/null; then
        INSTALL_METHOD="system_existing"
        PYTHON_TARGET="$(command -v python3)"
        echo -e "${C_GREEN}  ✓ Detected existing aiohttp in system Python.${C_RESET}"
    fi
fi

# If all installation methods failed, provide distro-specific resolution instructions
if [ -z "${INSTALL_METHOD}" ]; then
    echo -e "${C_RED}Error: Unable to install Fold dependencies (aiohttp).${C_RESET}"
    echo -e "Please install python3-venv or python3-aiohttp using your package manager:\n"
    if command -v apt-get >/dev/null 2>&1; then
        echo -e "  ${C_BOLD}sudo apt update && sudo apt install -y python3-venv python3-pip python3-aiohttp${C_RESET}"
    elif command -v dnf >/dev/null 2>&1; then
        echo -e "  ${C_BOLD}sudo dnf install -y python3-pip python3-aiohttp${C_RESET}"
    elif command -v pacman >/dev/null 2>&1; then
        echo -e "  ${C_BOLD}sudo pacman -S python-pip python-aiohttp${C_RESET}"
    elif command -v zypper >/dev/null 2>&1; then
        echo -e "  ${C_BOLD}sudo zypper install -y python3-pip python3-aiohttp${C_RESET}"
    elif command -v apk >/dev/null 2>&1; then
        echo -e "  ${C_BOLD}sudo apk add py3-pip py3-aiohttp py3-virtualenv${C_RESET}"
    else
        echo -e "  Install python3-venv or pip, then rerun ./install.sh"
    fi
    echo -e "\nAfter installing, rerun: ${C_BOLD}./install.sh${C_RESET}"
    exit 1
fi

# Generate the executable launcher in ~/.local/bin/fold
cat << INNER_EOF > "${LOCAL_BIN}/fold"
#!/usr/bin/env bash
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH:-}"
exec "${PYTHON_TARGET}" -m fold.cli "\$@"
INNER_EOF
chmod +x "${LOCAL_BIN}/fold"

# Generate backward-compatibility alias ~/.local/bin/sheprd
cat << INNER_EOF > "${LOCAL_BIN}/sheprd"
#!/usr/bin/env bash
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH:-}"
exec "${PYTHON_TARGET}" -m fold.cli "\$@"
INNER_EOF
chmod +x "${LOCAL_BIN}/sheprd"

# Verify the installed CLI runs cleanly
if ! "${LOCAL_BIN}/fold" --help >/dev/null 2>&1; then
    echo -e "${C_RED}Error: Fold CLI self-test failed after installation.${C_RESET}"
    exit 1
fi
echo -e "${C_GREEN}  ✓ Verified fold CLI operational.${C_RESET}"

# 5. Check Herdr Integration & Shell PATH
echo -e "${C_DIM}[5/5] Checking environment & Herdr workspace integration...${C_RESET}"
if command -v herdr >/dev/null 2>&1; then
    echo -e "${C_GREEN}  ✓ Herdr detected on PATH.${C_RESET}"
else
    echo -e "${C_DIM}  Note: Herdr workspace manager not found on PATH. (Optional; install from https://herdr.dev)${C_RESET}"
fi

# Ensure ~/.local/bin is configured in shell startup files
PATH_CONFIGURED=true
if [[ ":$PATH:" != *":${LOCAL_BIN}:"* ]]; then
    PATH_CONFIGURED=false
    SHELL_RC=""
    if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
        SHELL_RC="${HOME}/.zshrc"
    elif [ -f "${HOME}/.bashrc" ]; then
        SHELL_RC="${HOME}/.bashrc"
    fi

    if [ -n "${SHELL_RC}" ] && [ -f "${SHELL_RC}" ]; then
        if ! grep -q 'alias fold=' "${SHELL_RC}"; then
            echo '' >> "${SHELL_RC}"
            echo '# Added by Fold installer' >> "${SHELL_RC}"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${SHELL_RC}"
            echo 'alias fold="$HOME/.local/bin/fold"' >> "${SHELL_RC}"
            echo 'alias sheprd="$HOME/.local/bin/sheprd"' >> "${SHELL_RC}"
            echo -e "${C_GREEN}  ✓ Added Fold aliases & PATH to ${SHELL_RC}.${C_RESET}"
            PATH_CONFIGURED=true
        fi
    fi
fi

echo -e "\n${C_BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo -e "${C_BRIGHT_BLUE}✓ Fold installation complete!${C_RESET}"
if [ "${PATH_CONFIGURED}" = false ]; then
    echo -e "${C_AMBER}Make sure ${LOCAL_BIN} is in your PATH:${C_RESET}"
    echo -e "  ${C_BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${C_RESET}"
fi
echo -e "\n${C_BOLD}Quick Commands:${C_RESET}"
echo -e "  ${C_BLUE}fold cli${C_RESET}                 # Launch interactive Fold CLI console shell"
echo -e "  ${C_BLUE}fold web${C_RESET}                 # Launch the Fold Web UI"
echo -e "  ${C_BLUE}fold download qwen2.5-0.5b${C_RESET} # Download starter model (469MB)"
echo -e "  ${C_BLUE}fold list${C_RESET}                # View configured agents and ports"
echo -e "  ${C_BLUE}fold inspect <file.gguf>${C_RESET} # Auto-calculate optimal GPU/CPU settings"
echo -e "  ${C_BLUE}fold chat <name>${C_RESET}         # Interactive terminal chat"
echo -e "${C_BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}\n"
