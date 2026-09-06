# ⚡ Sheprd: llama.cpp Agent Orchestrator & Herdr Integration

> **Digital Green terminal & web orchestrator for deploying local llama.cpp models, spinning up specialized AI agents, and seamlessly spawning them inside Herdr terminal workspaces and Telegram.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-00ff66.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff66.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![llama.cpp: Vulkan](https://img.shields.io/badge/llama.cpp-Vulkan%20Accelerated-00ff66.svg?style=flat-square)](https://github.com/ggml-org/llama.cpp)
[![Herdr: Integrated](https://img.shields.io/badge/Herdr-Native%20Workspace-00ff66.svg?style=flat-square)](https://github.com/herdr/herdr)

---

## 🖥️ Screenshots

### Web UI Dashboard & Agent Fleet
![Sheprd Dashboard](docs/assets/sheprd_dashboard.png)

### Digital Green Interactive Terminal Chat (Herdr Integration)
![Sheprd Interactive Terminal](docs/assets/sheprd_terminal.png)

### Model Auto-Inspector & Hardware Tuning Wizard
![Sheprd Web UI Deployment Modal](docs/assets/sheprd_web_ui.png)

---

## 🚀 One-Step Linux Installation

Sheprd provides an automated setup script that configures your directories, checks Python dependencies, downloads the pre-built `llama.cpp` Vulkan acceleration bundle, and registers the `sheprd` CLI in `~/.local/bin`.

### Option 1: Quick Install (Git Clone)

```bash
git clone https://github.com/saxonmurray85-ops/sheprd.git
cd sheprd
chmod +x install.sh
./install.sh
```

Ensure `~/.local/bin` is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Option 2: Python Package (pip)

```bash
cd sheprd
pip install -e .
```

---

## ⚡ Quick Start

### 1. Download a Curated Starter Model

Sheprd includes a built-in streaming model downloader that automatically verifies GGUF headers:

```bash
sheprd download qwen2.5-0.5b
```

Available curated starter models:
- `qwen2.5-0.5b` (469 MB) — Ultra-fast, lightweight general assistant & testing
- `smollm2-135m` (98 MB) — Ultra-compact test model
- `llama-3.2-1b` (1.3 GB) — Compact meta model
- `qwen2.5-coder-1.5b` (1.1 GB) — Specialized coding agent

### 2. Inspect Any GGUF Model

Sheprd reads binary headers in milliseconds to determine layer count, context length, architecture, and optimal GPU offload:

```bash
sheprd inspect ~/.local/share/sheprd/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 3. Launch the Digital Green Web UI

```bash
sheprd web
# Or: sheprd ui
```
Navigate to: **`http://127.0.0.1:8765`**

### 4. Chat Inside Herdr Terminal

Once an agent is deployed (e.g. `sage`), launch it in Herdr by typing its name in any terminal pane:

```bash
sage
```
Or trigger automatic tab creation from the CLI or Web UI:
```bash
sheprd spawn sage
```

---

## 🧠 Key Features

### 1. Pure-Python GGUF Auto-Inspection
- Reads GGUF v2/v3 binary structures without third-party dependencies.
- Extracts architecture (`llama`, `qwen2`, `mistral`, `gemma`, `phi3`), quantization format, layer count, context window size, and chat template strings.

### 2. Hardware-Aware Optimal Configuration
- Automatically calculates CPU physical/logical threads and detects discrete GPUs (AMD Radeon RX 7900 XTX via Vulkan, NVIDIA via CUDA).
- Calculates the optimal `--n-gpu-layers` based on available VRAM and parameter sizes.
- Prevents resource exhaustion by sizing batch sizes and context memory to physical limits.

### 3. Native Herdr Workspace Integration
- **CLI Executable**: Automatically creates an executable launcher at `~/.local/bin/<agent_name>`.
- **Typing the Agent Name**: In any Herdr pane, typing `<agent_name>` instantly connects to the local llama-server, streams responses in real-time, and manages Herdr pane titles and status indicators (`working` / `idle`).
- **One-Click Tab Spawning**: Web UI and CLI communicate over Herdr's Unix domain socket (`~/.config/herdr/herdr.sock`) using `herdr tab create --label <name>` and `herdr pane run`.

### 4. Telegram Bot Support
- Connect any agent directly to a Telegram bot by providing your bot token in the Web UI or CLI.
- Asynchronous polling worker dispatches incoming messages with system persona prompts and returns formatted markdown.
- Token scrubbing and chat allowlists (`SHEPRD_TELEGRAM_ALLOWED_USERS`) ensure secure remote interaction.

### 5. Multi-Agent Swarms & Callable Permissions
- **`callable_by_agents` (True/False)**: Enforces boundary controls. Private agents cannot be summoned by peer agents.
- **Group Swarms**: Tag agents into groups (`dev`, `research`, `triage`). Send broadcast prompts to execute tasks across all agents in the group.
- **Peer Delegation**: Call `@agent_name` inside interactive chat to route sub-prompts directly to specialized agents.
- **Agent-Hub Synchronization**: Automatically syncs agent metadata with `~/.agent-hub/data/memory.db` for multi-tool discovery.

---

## 🔒 Hardened Security Model

Sheprd implements defense-in-depth across all system boundaries:

| Control | Implementation |
|---|---|
| **Path Traversal Guard** | Canonical resolution via `os.path.realpath`. Rejects system paths (`/etc`, `/proc`, `/sys`, `/root`) and user credential stores (`~/.ssh`, `~/.gnupg`, `secrets.env`). |
| **GGUF Binary Validation** | Verifies `b'GGUF'` magic bytes before launching any binary process. |
| **Subprocess Isolation** | Executes `llama-server` strictly via `list[str]` arguments (no `shell=True`, no bash string interpolation). |
| **CSRF & API Token Defense** | Enforces Origin validation and `X-Sheprd-Token` checking for state-changing HTTP endpoints. Blocks cross-origin attacks. |
| **PID Verification Guard** | Inspects `/proc/{pid}/cmdline` before sending termination signals to verify the target process is actually `llama-server`. |
| **Strict File Permissions** | SQLite database and WAL files are maintained at `0600` (user read/write only). |
| **Token Scrubbing** | Telegram tokens are masked in API responses (`123456:••••••••••••xYZ`) and scrubbed from all server logs. |
| **Localhost Binding** | Servers and Web UI bind exclusively to `127.0.0.1`. |

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Sheprd Web UI (aiohttp)     │
                                  │   http://127.0.0.1:8765       │
                                  └───────────────┬───────────────┘
                                                  │
                      ┌───────────────────────────┼──────────────────────────┐
                      │                           │                          │
                      ▼                           ▼                          ▼
            ┌───────────────────┐       ┌───────────────────┐      ┌───────────────────┐
            │ GGUF Inspector    │       │ Server Manager    │      │ Telegram Workers  │
            │ & Auto-Config     │       │ (llama-server)    │      │ (Async Polling)   │
            └───────────────────┘       └─────────┬─────────┘      └─────────┬─────────┘
                                                  │                          │
                                                  ▼                          ▼
                                        ┌───────────────────┐      ┌───────────────────┐
                                        │ Local llama-server│◄─────┤ Telegram Chat     │
                                        │ Port 8081..8999   │      └───────────────────┘
                                        └─────────▲─────────┘
                                                  │
                      ┌───────────────────────────┴──────────────────────────┐
                      │                                                      │
                      ▼                                                      ▼
            ┌───────────────────┐                                  ┌───────────────────┐
            │ Herdr Integration │                                  │ Agent-Hub Memory  │
            │ ~/.local/bin/name │                                  │ ~/.agent-hub/     │
            │ herdr.sock API    │                                  │ data/memory.db    │
            └───────────────────┘                                  └───────────────────┘
```

---

## ⌨️ CLI Reference

| Command | Description |
|---|---|
| `sheprd web [--port 8765]` | Launch the Digital Green Web UI |
| `sheprd list` | List all configured agents, ports, and statuses |
| `sheprd inspect <path>` | Auto-inspect GGUF file and display hardware recommendations |
| `sheprd download [model_key]` | Download starter models with progress indicator |
| `sheprd start <name>` | Start the llama-server and Telegram bot for an agent |
| `sheprd stop <name>` | Stop an agent's server process |
| `sheprd stop-all` | Stop all active agent servers |
| `sheprd spawn <name>` | Spawn agent in an active Herdr terminal tab |
| `sheprd chat <name>` | Launch interactive Digital Green terminal chat |
| `sheprd logs <name>` | View recent server logs |
| `sheprd remove <name>` | Delete an agent and uninstall its Herdr launcher |

---

## 🧪 Testing

Sheprd includes a comprehensive unit test suite covering security guards, GGUF binary extraction, database persistence, group validation, and multi-agent loops:

```bash
python3 -m unittest tests/test_sheprd.py -v
```

---

## 📄 License

MIT License. Designed and built for seamless pairing with Herdr and local open-source LLMs.
