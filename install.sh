#!/usr/bin/env bash
# ==============================================================================
# Sheprd: One-Line Linux Installer & Setup Script
# Installs llama.cpp Vulkan runtime, Sheprd CLI, and environment configuration.
# ==============================================================================

set -euo pipefail

# ANSI Digital Green Styling
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_GREEN="\033[38;5;48m"
C_BRIGHT="\033[38;5;84m"
C_DIM="\033[2m"
C_CYAN="\033[38;5;51m"
C_RED="\033[38;5;196m"

echo -e "${C_BRIGHT}"
cat << 'EOF'
  ___ _  _ ___ ___ ___ ___ 
 / __| || | __| _ \ _ \   \ 
 \__ \ __ | _||  _/   / |) |
 |___/_||_|___|_| |_|_\___/ 
EOF
echo -e "${C_RESET}"
echo -e "${C_GREEN}${C_BOLD}Installing Sheprd // llama.cpp Agent Orchestrator...${C_RESET}\n"

# 1. Dependency checks
echo -e "${C_DIM}[1/5] Checking prerequisites...${C_RESET}"
command -v python3 >/dev/null 2>&1 || { echo -e "${C_RED}Error: python3 is required.${C_RESET}"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo -e "${C_RED}Error: curl is required.${C_RESET}"; exit 1; }
command -v tar >/dev/null 2>&1 || { echo -e "${C_RED}Error: tar is required.${C_RESET}"; exit 1; }

# Verify Python version >= 3.10
python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo -e "${C_RED}Error: Python 3.10 or higher is required.${C_RESET}"
    exit 1
}

# 2. Directory structure
echo -e "${C_DIM}[2/5] Creating directory layout...${C_RESET}"
SHEPRD_HOME="${HOME}/.local/share/sheprd"
SHEPRD_BIN="${SHEPRD_HOME}/bin"
SHEPRD_MODELS="${SHEPRD_HOME}/models"
SHEPRD_LOGS="${SHEPRD_HOME}/logs"
LOCAL_BIN="${HOME}/.local/bin"

mkdir -p "${SHEPRD_BIN}" "${SHEPRD_MODELS}" "${SHEPRD_LOGS}" "${LOCAL_BIN}" "${HOME}/.config/sheprd"
chmod 700 "${SHEPRD_HOME}" "${HOME}/.config/sheprd" || true

# 3. Download llama.cpp Vulkan runtime if not present
echo -e "${C_DIM}[3/5] Checking llama.cpp binary bundle...${C_RESET}"
if [ ! -f "${SHEPRD_BIN}/llama-server" ]; then
    echo -e "${C_CYAN}  Downloading prebuilt llama.cpp (Vulkan/x86_64)...${C_RESET}"
    LLAMA_RELEASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/b10827/llama-b10827-bin-ubuntu-vulkan-x64.tar.gz"
    EXPECTED_SHA256="9005df90e98f94bbf20b328104e4481a1a09fbc47c3039617923ffb1287fdcc4"
    TEMP_TAR="/tmp/sheprd-llama-$$.tar.gz"
    
    if curl -sSL -o "${TEMP_TAR}" "${LLAMA_RELEASE_URL}"; then
        ACTUAL_SHA256="$(sha256sum "${TEMP_TAR}" | awk '{print $1}')"
        if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
            echo -e "${C_RED}  Error: SHA-256 checksum mismatch for llama.cpp binary bundle!${C_RESET}"
            echo -e "${C_RED}  Expected: ${EXPECTED_SHA256}${C_RESET}"
            echo -e "${C_RED}  Got:      ${ACTUAL_SHA256}${C_RESET}"
            rm -f "${TEMP_TAR}"
            exit 1
        fi
        tar -xzf "${TEMP_TAR}" -C "${SHEPRD_BIN}" --strip-components=1
        rm -f "${TEMP_TAR}"
        echo -e "${C_GREEN}  llama-server runtime verified and installed to ${SHEPRD_BIN}.${C_RESET}"
    else
        echo -e "${C_RED}  Failed to download llama.cpp release. You can place your own llama-server in ${SHEPRD_BIN}.${C_RESET}"
    fi
else
    echo -e "${C_GREEN}  llama-server runtime is already present.${C_RESET}"
fi

# 4. Install Sheprd package & CLI
echo -e "${C_DIM}[4/5] Installing Sheprd package and registering CLI...${C_RESET}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install package in editable user mode
if python3 -m pip install --user -e "${SCRIPT_DIR}" >/dev/null 2>&1; then
    echo -e "${C_GREEN}  Installed Sheprd via pip in user mode.${C_RESET}"
else
    echo -e "${C_DIM}  pip install failed or unavailable; falling back to direct wrapper script.${C_RESET}"
    cat << EOF > "${LOCAL_BIN}/sheprd"
#!/usr/bin/env bash
PYTHON_BIN="\$(command -v python3 2>/dev/null || echo "python3")"
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH:-}"
exec "\${PYTHON_BIN}" -m sheprd.cli "\$@"
EOF
    chmod +x "${LOCAL_BIN}/sheprd"
fi

# Also symlink llama-server / llama-cli into ~/.local/bin if they exist
if [ -f "${SHEPRD_BIN}/llama-server" ]; then
    cat << EOF > "${LOCAL_BIN}/llama-server"
#!/usr/bin/env bash
exec "${SHEPRD_BIN}/llama-server" "\$@"
EOF
    chmod +x "${LOCAL_BIN}/llama-server"
fi

if [ -f "${SHEPRD_BIN}/llama-cli" ]; then
    cat << EOF > "${LOCAL_BIN}/llama-cli"
#!/usr/bin/env bash
exec "${SHEPRD_BIN}/llama-cli" "\$@"
EOF
    chmod +x "${LOCAL_BIN}/llama-cli"
fi

# 5. Verify Herdr status
echo -e "${C_DIM}[5/5] Checking Herdr integration...${C_RESET}"
if command -v herdr >/dev/null 2>&1; then
    echo -e "${C_GREEN}  ✓ Herdr detected on PATH.${C_RESET}"
else
    echo -e "${C_DIM}  Note: Herdr workspace manager not found on PATH. (Optional; install from https://herdr.dev)${C_RESET}"
fi

echo -e "\n${C_BRIGHT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo -e "${C_BRIGHT}✓ Installation complete!${C_RESET}"
echo -e "${C_GREEN}Make sure ${LOCAL_BIN} is in your PATH:${C_RESET}"
echo -e "  ${C_BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${C_RESET}"
echo -e "\n${C_BOLD}Quick Commands:${C_RESET}"
echo -e "  ${C_CYAN}sheprd web${C_RESET}                 # Launch the Digital Green Web UI"
echo -e "  ${C_CYAN}sheprd download qwen2.5-0.5b${C_RESET} # Download starter model (469MB)"
echo -e "  ${C_CYAN}sheprd list${C_RESET}                # View configured agents and ports"
echo -e "  ${C_CYAN}sheprd inspect <file.gguf>${C_RESET} # Auto-calculate optimal GPU/CPU settings"
echo -e "${C_BRIGHT}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}\n"
