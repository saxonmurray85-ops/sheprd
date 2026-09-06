"""
Security module for Sheprd.
Provides rigorous path validation, model signature verification,
agent name sanitization, secret masking, and local-only network constraints.
"""

from pathlib import Path
import os
import re
from typing import List, Optional

# Regex for safe agent names: alphanumeric, dashes, underscores (1-32 chars)
AGENT_NAME_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}$")

# Disallowed system and sensitive directories
DISALLOWED_DIRECTORY_PREFIXES = [
    Path("/etc"),
    Path("/root"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/boot"),
    Path("/var/log"),
]

# Sensitive file or folder markers
SENSITIVE_PATH_MARKERS = [
    ".ssh",
    ".gnupg",
    "secrets.env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".bash_history",
    ".zsh_history",
    "shadow",
]

RESERVED_NAMES = {
    "sheprd",
    "herdr",
    "bin",
    "root",
    "daemon",
    "admin",
    "api",
    "system",
    "null",
    "default",
}

GGUF_MAGIC = b"GGUF"


class SecurityError(Exception):
    """Raised when a security validation constraint is violated."""
    pass


def validate_agent_name(name: str) -> str:
    """
    Validate agent name strictly to prevent shell injection, path traversal,
    and reserved command collisions.
    """
    name = (name or "").strip()
    if not name:
        raise SecurityError("Agent name cannot be empty.")
    if len(name) > 32:
        raise SecurityError("Agent name must be 32 characters or fewer.")
    if not AGENT_NAME_REGEX.match(name):
        raise SecurityError(
            f"Invalid agent name '{name}'. Must start with alphanumeric and contain only letters, numbers, hyphens, and underscores."
        )
    if name.lower() in RESERVED_NAMES:
        raise SecurityError(f"'{name}' is a reserved name and cannot be used as an agent name.")
    return name


def validate_model_path(path_str: str) -> Path:
    """
    Validate that model path is a safe, existing, readable GGUF file
    without path traversal or access to sensitive system files.
    """
    if not path_str or not isinstance(path_str, str):
        raise SecurityError("Model path must be a non-empty string.")

    cleaned = path_str.strip()
    # Expand user directory (~/...) and environment variables
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    target = Path(expanded).resolve()

    # Verify not root or system directory (S1 fix: never catch SecurityError)
    for disallowed in DISALLOWED_DIRECTORY_PREFIXES:
        try:
            resolved_disallowed = disallowed.resolve()
            if target == resolved_disallowed or resolved_disallowed in target.parents or target.is_relative_to(resolved_disallowed):
                raise SecurityError(f"Access to system directory '{disallowed}' is forbidden.")
        except SecurityError:
            raise
        except (ValueError, OSError):
            pass

    # Verify not containing sensitive markers
    target_parts = target.parts
    for marker in SENSITIVE_PATH_MARKERS:
        if marker in target_parts or any(marker in part for part in target_parts):
            raise SecurityError(f"Access to sensitive path containing '{marker}' is prohibited.")

    if not target.exists():
        raise SecurityError(f"Model file does not exist at '{target}'.")

    if not target.is_file():
        raise SecurityError(f"Target path '{target}' is not a regular file.")

    if not os.access(target, os.R_OK):
        raise SecurityError(f"Model file at '{target}' is not readable (check permissions).")

    # Verify GGUF magic header bytes
    try:
        with open(target, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise SecurityError(
                    f"File '{target.name}' is not a valid GGUF file. Magic bytes were {magic!r}, expected {GGUF_MAGIC!r}."
                )
    except OSError as e:
        raise SecurityError(f"Failed to inspect model header: {e}")

    return target


def mask_token(token: Optional[str]) -> str:
    """
    Safely mask a secret token for display in UI or logs.
    Example: 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ -> 123456:••••••••••••xYZ
    """
    if not token:
        return ""
    token = token.strip()
    if len(token) <= 8:
        return "••••••••"
    prefix = token[:6]
    suffix = token[-3:]
    return f"{prefix}:••••••••••••{suffix}"


def sanitize_log_text(text: str, tokens_to_mask: Optional[List[str]] = None) -> str:
    """
    Scrub tokens and secrets from log text before writing or streaming.
    """
    if not text:
        return ""
    sanitized = text
    if tokens_to_mask:
        for tok in tokens_to_mask:
            if tok and len(tok) > 4:
                sanitized = sanitized.replace(tok, mask_token(tok))
    # General Telegram token pattern: \d{8,12}:[A-Za-z0-9_-]{35}
    sanitized = re.sub(
        r"(\b\d{8,12}:)[A-Za-z0-9_-]{30,45}\b",
        r"\1••••••••••••••••",
        sanitized
    )
    return sanitized


def validate_port(port: int) -> int:
    """
    Validate that port is within safe unprivileged non-system port range.
    """
    try:
        p = int(port)
    except (ValueError, TypeError):
        raise SecurityError(f"Port '{port}' must be a valid integer.")

    if not (1024 <= p <= 65535):
        raise SecurityError(f"Port {p} is out of safe range (1024-65535).")
    return p


def validate_context_size(ctx: int) -> int:
    """Validate context window size is within sane limits."""
    try:
        c = int(ctx)
    except (ValueError, TypeError):
        raise SecurityError("Context size must be an integer.")
    if not (256 <= c <= 262144):
        raise SecurityError(f"Context size {c} is out of safe range (256 - 262,144).")
    return c


def validate_gpu_layers(layers: int) -> int:
    """Validate GPU layer count."""
    try:
        l = int(layers)
    except (ValueError, TypeError):
        raise SecurityError("GPU layer count must be an integer.")
    if not (0 <= l <= 512):
        raise SecurityError(f"GPU layers {l} is out of safe range (0 - 512).")
    return l


GROUP_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


def validate_group_name(group: str) -> str:
    """Validate group name (S7)."""
    group = (group or "").strip()
    if not group:
        raise SecurityError("Group name cannot be empty.")
    if not GROUP_NAME_REGEX.match(group):
        raise SecurityError(
            f"Invalid group name '{group}'. Must contain only letters, numbers, hyphens, and underscores (max 32 chars)."
        )
    return group


def validate_groups(groups: Optional[List[str]]) -> List[str]:
    """Validate list of group names (S7)."""
    if not groups:
        return ["default"]
    valid = []
    for g in groups:
        clean = validate_group_name(g)
        if clean not in valid:
            valid.append(clean)
    return valid or ["default"]

