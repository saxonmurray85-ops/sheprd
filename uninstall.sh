#!/usr/bin/env bash
# ==============================================================================
# Fold: Linux Uninstaller
# Removes Fold binaries, environments, databases, and logs.
# ==============================================================================

set -euo pipefail

C_RESET="\033[0m"
C_BOLD="\033[1m"
C_BLUE="\033[38;5;117m"
C_DIM="\033[2m"
C_RED="\033[38;5;217m"

echo -e "${C_BOLD}Uninstalling Fold...${C_RESET}"

# Stop any running fold servers
if command -v fold >/dev/null 2>&1; then
    fold stop-all 2>/dev/null || true
elif [ -x "${HOME}/.local/bin/fold" ]; then
    "${HOME}/.local/bin/fold" stop-all 2>/dev/null || true
fi

# Remove CLI launchers
rm -f "${HOME}/.local/bin/fold" "${HOME}/.local/bin/sheprd"

# Remove application data, venv, binaries, and logs
rm -rf "${HOME}/.local/share/fold"

# Remove config
rm -rf "${HOME}/.config/fold"

echo -e "${C_BLUE}✓ Fold has been completely uninstalled.${C_RESET}"
