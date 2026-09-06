#!/usr/bin/env bash
# ==============================================================================
# Sheprd: Linux Uninstaller
# Removes Sheprd binaries, environments, databases, and logs.
# ==============================================================================

set -euo pipefail

C_RESET="\033[0m"
C_BOLD="\033[1m"
C_GREEN="\033[38;5;48m"
C_DIM="\033[2m"
C_RED="\033[38;5;196m"

echo -e "${C_BOLD}Uninstalling Sheprd...${C_RESET}"

# Stop any running sheprd servers if sheprd is currently callable
if command -v sheprd >/dev/null 2>&1; then
    sheprd stop-all 2>/dev/null || true
elif [ -x "${HOME}/.local/bin/sheprd" ]; then
    "${HOME}/.local/bin/sheprd" stop-all 2>/dev/null || true
fi

# Remove CLI launcher
rm -f "${HOME}/.local/bin/sheprd"

# Remove application data, venv, binaries, and logs
rm -rf "${HOME}/.local/share/sheprd"

# Remove config
rm -rf "${HOME}/.config/sheprd"

echo -e "${C_GREEN}✓ Sheprd has been completely uninstalled.${C_RESET}"
