# 🐑 Fold: llama.cpp Agent Orchestrator & Herdr Integration

> **Terminal & web orchestrator for deploying local llama.cpp models, spinning up specialized AI agents, and seamlessly spawning them inside Herdr terminal workspaces and Telegram.**

<p align="center">
  <img src="docs/assets/fold_trio.png" alt="Fold Origami Sheep Branding" width="600" style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);" />
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-7eb8da.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-88b79b.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![llama.cpp: Vulkan / ROCm](https://img.shields.io/badge/llama.cpp-Vulkan%20%2F%20ROCm-cbd5e1.svg?style=flat-square)](https://github.com/ggml-org/llama.cpp)
[![Herdr: Integrated](https://img.shields.io/badge/Herdr-Native%20Workspace-7eb8da.svg?style=flat-square)](https://github.com/herdr/herdr)

---

## 🖥️ Screenshots

### Fold Pastel Web UI & Live Agent Fleet
![Fold Web UI](docs/assets/fold_web_ui.png)

### Model Auto-Inspector & Hardware Tuning Wizard (Gemma-4 & Quantized Support)
![Fold Deploy Wizard](docs/assets/fold_dashboard.png)

### Herdr Terminal Chat & Pure-Python Model Inspection
![Fold Terminal Chat](docs/assets/fold_terminal.png)

---

## 🎨 Aesthetics & Branding: Origami Sheep & Pastel Palette

Fold introduces an elegant, modern visual identity inspired by origami papercraft and a soft pastel aesthetic:
- **Origami Paper Sheep**: Distinct faceted geometric origami sheep avatars for your agent fleet, status badges, and header branding.
- **Pastel Color System**: Carefully calibrated tones of **Sky Blue** (`#7eb8da`), **Sage Green** (`#88b79b`), and **Slate Grey** (`#cbd5e1` / `#e2e8f0`) on a light, airy paper canvas with crisp cards and origami papercraft accents.
- **Clean Responsive Layout**: Identical workflow placement with rounded modern cards, subtle ambient glows, and clean typography.

<p align="center">
  <img src="docs/assets/sheep_sage.png" width="115" alt="Sage Green Origami Sheep" />
  <img src="docs/assets/sheep_blue.png" width="115" alt="Sky Blue Origami Sheep" />
  <img src="docs/assets/sheep_cream.png" width="115" alt="Cream Origami Sheep" />
  <img src="docs/assets/sheep_pink.png" width="115" alt="Pastel Pink Origami Sheep" />
</p>

---

## 🚀 One-Step Linux Installation

Fold provides an automated setup script that configures an isolated environment, automatically resolves Python dependencies across all major Linux distributions (Ubuntu, Debian, Fedora, Arch, openSUSE, Alpine), downloads the pre-built `llama.cpp` Vulkan acceleration bundle, and registers the `fold` CLI in `~/.local/bin`.

### Option 1: Quick Install (Git Clone)

```bash
git clone https://github.com/saxonmurray85-ops/sheprd.git
cd sheprd
chmod +x install.sh
./install.sh
```

The installer automatically handles PEP 668 managed environments (Ubuntu 24.04+, Debian 12+, Fedora, Arch) using an isolated virtual environment and configures `~/.local/bin` in your shell startup configuration.

### Option 2: Python Package (pip)

```bash
cd sheprd
pip install -e .
```

### Uninstallation

To completely remove Fold, run:
```bash
./uninstall.sh
```

---

## ⚡ Quick Start

### 1. Download a Curated Starter Model

Fold includes a built-in streaming model downloader that automatically verifies GGUF headers:

```bash
fold download qwen2.5-0.5b
```

Available curated starter models:
- `qwen2.5-0.5b` (469 MB) — Ultra-fast, lightweight general assistant & testing
- `smollm2-135m` (98 MB) — Ultra-compact test model
- `llama-3.2-1b` (1.3 GB) — Compact meta model
- `qwen2.5-coder-1.5b` (1.1 GB) — Specialized coding agent

> [!TIP]
> **Recommended Models for Autonomous Tool Calling:**
> While compact models (<1.5B) run fast on low-spec hardware, reliable tool calling (invoking weather, search, calculations, and MCP plugins) requires models with **3B+ parameters** tuned for instruction following and function calling (such as `Qwen2.5-3B-Instruct`, `Llama-3.2-3B-Instruct`, `Mistral-7B-Instruct`, or `Qwen2.5-Coder-7B`). Models below 1.5B often lack the structural adherence to reliably generate valid JSON tool calls.

### 2. Inspect Any GGUF Model

Fold reads binary headers in milliseconds to determine layer count, context length, architecture, and optimal GPU offload:

```bash
fold inspect ~/.local/share/fold/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 3. Launch the Fold Web UI

```bash
fold web
# Or: fold ui
```
Navigate to: **`http://127.0.0.1:8765`**

### 4. Chat Inside Herdr Terminal

Once an agent is deployed (e.g. `sage`), launch it in Herdr by typing its name in any terminal pane:

```bash
sage
```
Or trigger automatic tab creation from the CLI or Web UI:
```bash
fold spawn sage
```

---

## 🧠 Key Features

### 1. Pure-Python GGUF Auto-Inspection & Multi-Architecture Engine
- Reads GGUF v2/v3 binary structures without third-party dependencies in milliseconds.
- Robust multi-architecture support: `gemma4`, `gemma3`, `gemma2`, `llama`, `qwen2`, `mistral`, `devstral`, `glm`, `deepseek`, `phi3`.
- Safely parses sliding-window attention and variable per-layer KV head arrays (`head_count_kv = [8, 8, ...]`) found in modern Gemma-4 and Gemma-2 architectures, calculating exact KV cache memory footprints.
- Automatically discovers chat template formats (`gemma4`, `llama3`, `chatml`, `mistral`, `deepseek`) and applies `--jinja` flag.

### 2. Advanced Quantization & Sharded Model Architecture
- **Expanded Quant Support**: Full compatibility with modern quantization types including `MXFP4`, `IQ4_NL`, `IQ4_XS`, `IQ3_XXS`, `UD-Q4_K_XL`, `TQ1_0`, `TQ2_0`, `Q4_0_4_4`, and standard K-quants (`Q4_K_M`, `Q8_0`).
- **Multi-Part / Sharded GGUF Discovery**: Automatically discovers and aggregates multi-shard model files (`model-00001-of-0000X.gguf`), computing cumulative parameter weight and total VRAM requirements across all parts.
- **Hardware-Aware Optimal Offloading**: Calculates CPU physical/logical threads and detects discrete GPUs (AMD Radeon RX 7900 XTX / 7900 GRE via Vulkan & ROCm, NVIDIA via CUDA). Automatically determines optimal `--n-gpu-layers` and batch sizing.
- **Dynamic Startup Timeout Scaling**: Automatically scales initialization timeouts up to 300 seconds for massive models (>70GB) to prevent premature timeouts during weight verification.
- **Large-Model Memory Tuning**: Automatically applies `--no-mmap` for massive models (>40GB) or unified memory setups to prevent system thrashing.

### 3. Native Herdr Workspace Integration
- **CLI Executable**: Automatically creates an executable launcher at `~/.local/bin/<agent_name>`.
- **Typing the Agent Name**: In any Herdr pane, typing `<agent_name>` instantly connects to the local llama-server, streams responses in real-time, and manages Herdr pane titles and status indicators (`working` / `idle`).
- **One-Click Tab Spawning**: Web UI and CLI communicate over Herdr's Unix domain socket (`~/.config/herdr/herdr.sock`) using `herdr tab create --label <name>` and `herdr pane run`.

### 4. Telegram Bot Support
- Connect any agent directly to a Telegram bot by providing your bot token in the Web UI or CLI.
- Asynchronous polling worker dispatches incoming messages with system persona prompts and returns formatted markdown.
- Token scrubbing and chat allowlists (`FOLD_TELEGRAM_ALLOWED_USERS`) ensure secure remote interaction.

### 5. Multi-Agent Swarms & Callable Permissions
- **`callable_by_agents` (True/False)**: Enforces boundary controls. Private agents cannot be summoned by peer agents.
- **Group Swarms**: Tag agents into groups (`dev`, `research`, `triage`). Send broadcast prompts to execute tasks across all agents in the group.
- **Peer Delegation**: Call `@agent_name` inside interactive chat to route sub-prompts directly to specialized agents.
- **Agent-Hub Synchronization**: Automatically syncs agent metadata with `~/.agent-hub/data/memory.db` under the `fold` namespace for multi-tool discovery.

### 6. Tools & Skills Ecosystem (Core + MCP Hub)
Fold equips your local agents with live autonomous tool execution using standard OpenAI function calling format supported natively by `llama-server`:

- **⚡ Built-in Core Tools (Zero-Setup & Zero-Dependency)**:
  - 🌦️ **`get_weather`**: Real-time live weather conditions, temperature, humidity, and wind via `wttr.in`.
  - 📖 **`wikipedia_search`**: Fact lookups, article summaries, and related topics via Wikipedia REST API.
  - 🌐 **`fetch_url`**: Read and extract clean text from public web documentation or articles.
  - 🧮 **`calculate`**: Safe AST-based mathematical expression evaluator (arithmetic, trigonometry, logarithms, roots).
  - ⏱️ **`get_current_time`**: Live UTC time, system local timezone, and UNIX timestamps.
- **🔌 Model Context Protocol (MCP) Client & 1-Click Smithery Hub**:
  - **1-Click Smithery & Registry Install**: Paste any Smithery tool URL (e.g. `https://smithery.ai/server/@smithery-ai/fetch`), package identifier (`@smithery-ai/...`), or terminal command. Fold automatically parses, installs, and connects it.
  - **Zero-Config Popular App Store**: 1-click zero-setup additions directly from the Web UI for **Web Page Fetcher** (`uvx mcp-server-fetch`), **SQLite Database Inspector** (`uvx mcp-server-sqlite`), **Memory Graph** (`npx -y @modelcontextprotocol/server-memory`), **Git Inspector** (`uvx mcp-server-git`), **Local Filesystem**, and **Brave Search**.
  - **Seamless Bi-directional Claude & Cursor Sync**: Any tool installed via Smithery Web (*"Install with Claude"*) or `smithery install --client claude` automatically synchronizes into Fold in real time!
  - **Live Smithery Search**: Query over 100K+ Smithery MCP tools directly from the Web UI or CLI.
  - Available across Web Chat, Herdr interactive terminal, and Telegram bots.
  - Automatic tool selection, iterative execution loops, and error recovery.

### 7. Dynamic LRU Model Hot-Swapping
Running multiple local LLM servers simultaneously quickly exhausts GPU VRAM and system memory. Fold includes an intelligent Least-Recently-Used (LRU) model hot-swapping engine:
- **Zero-Friction Activation**: When an agent is called (via Web Chat, Herdr terminal `fold spawn`, `@mention`, or Telegram), Fold ensures its server is active.
- **Configurable VRAM Ceiling (`FOLD_MAX_ACTIVE_MODELS`)**: By default, Fold keeps `1` active model loaded in VRAM (perfect for single-GPU or shared systems). When a new agent is invoked, the least-recently-used server is automatically stopped and evicted before the new model loads.
- **Multi-GPU Scalability**: Set `export FOLD_MAX_ACTIVE_MODELS=3` (or any integer) to support multiple concurrent active models on larger rigs.

---

## 🔒 Hardened Security Model

Fold implements defense-in-depth across all system boundaries:

| Control | Implementation |
|---|---|
| **Path Traversal Guard** | Canonical resolution via `os.path.realpath`. Rejects system paths (`/etc`, `/proc`, `/sys`, `/root`) and user credential stores (`~/.ssh`, `~/.gnupg`, `secrets.env`). |
| **SSRF & Network Shield** | `fetch_url` verifies DNS resolutions and enforces strict IP-level rejection of RFC-1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), and cloud metadata services (`169.254.169.254`), with recursive redirect tracking and a 64KB download ceiling. |
| **AST Exponent Bomb Guard** | Math evaluator rejects malicious nesting and caps exponents (`abs(exp) <= 100`, `abs(base) <= 10000`), neutralizing algorithmic complexity attacks like `9**9**9`. |
| **GGUF Binary Validation** | Verifies `b'GGUF'` magic bytes before launching any binary process. |
| **Subprocess Isolation** | Executes `llama-server` strictly via `list[str]` arguments (no `shell=True`, no bash string interpolation). |
| **CSRF & API Token Defense** | Enforces Origin validation and unconditional `X-Fold-Token` (and `X-Sheprd-Token`) checking for state-changing HTTP endpoints. API token persisted to `~/.local/share/fold/api_token` with `0600` permissions. |
| **PID Verification Guard** | Inspects `/proc/{pid}/cmdline` before sending termination signals to verify the target process is actually `llama-server`. |
| **Strict File Permissions** | SQLite database and WAL files are maintained at `0600` (user read/write only). |
| **Token Scrubbing** | Telegram tokens and MCP environment secrets are masked in API responses (`••••`) and scrubbed from all server logs. |
| **Localhost Binding** | Servers and Web UI bind exclusively to `127.0.0.1`. |

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Fold Web UI (aiohttp)       │
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
                                        │ Port 8081..8999   │      └─────────┬─────────┘
                                        └─────────▲─────────┘                │
                                                  │                          │
                      ┌───────────────────────────┴──────────────────────────┴┐
                      │                                                       │
                      ▼                                                       ▼
            ┌───────────────────┐                                   ┌───────────────────┐
            │ Herdr Integration │                                   │ Tool Hub & MCP    │
            │ ~/.local/bin/name │                                   │ JSON-RPC Client   │
            │ herdr.sock API    │                                   │ 🌦️ 📖 🧮 🔌 💾     │
            └───────────────────┘                                   └───────────────────┘
```

---

## ⌨️ CLI Reference

| Command | Description |
|---|---|
| `fold web [--port 8765]` | Launch the Fold Pastel Web UI & background services |
| `fold list` | List all configured agents, ports, and statuses |
| `fold inspect <path>` | Auto-inspect GGUF file and display hardware recommendations |
| `fold download [model_key]` | Download starter models with progress indicator |
| `fold start <name>` | Start the llama-server and Telegram bot for an agent |
| `fold activate <name>` | Hot-swap/load agent model into memory (LRU eviction if at limit) |
| `fold stop <name>` | Stop an agent's server process |
| `fold stop-all` | Stop all active agent servers |
| `fold spawn <name>` | Spawn agent in an active Herdr terminal tab |
| `fold chat <name>` | Launch interactive terminal chat with live tool execution |
| `fold tools` | Display the system-wide catalog of Core and MCP tools |
| `fold mcp list` | List registered external MCP servers |
| `fold mcp install <target>` | 1-click install from Smithery URL, package identifier, or command |
| `fold mcp sync` | Synchronize MCP servers from Claude Desktop and Cursor configs |
| `fold mcp search <query>` | Search the Smithery 100K+ tool registry |
| `fold mcp add <name> <cmd> [args...]` | Register and connect a new stdio MCP server manually |
| `fold mcp remove <name>` | Disconnect and remove an MCP server |
| `fold telegram status` | Check connectivity status of all configured Telegram bots |
| `fold telegram run` | Run foreground standalone Telegram bot worker |
| `fold logs <name>` | View recent server logs |
| `fold remove <name>` | Delete an agent and uninstall its Herdr launcher |

*(Note: The `sheprd` CLI command is maintained as a transparent backward-compatible alias for all `fold` commands).*

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FOLD_MAX_ACTIVE_MODELS` | `1` | Maximum concurrent `llama-server` instances kept loaded in RAM/VRAM. When exceeded, the least recently used model is automatically stopped. Increase this on multi-GPU setups. *(Legacy fallback: `SHEPRD_MAX_ACTIVE_MODELS`)* |
| `FOLD_TELEGRAM_ALLOWED_USERS` | *(none / open)* | Comma-separated list of allowed Telegram usernames or user IDs (e.g. `user1,12345678`). Restricts remote bot access. *(Legacy fallback: `SHEPRD_TELEGRAM_ALLOWED_USERS`)* |
| `FOLD_HOME` | `~/.local/share/fold` | Root directory for Fold databases, logs, binaries, and lockfiles. |

---

## 🧪 Testing

Fold includes a comprehensive unit test suite covering security guards, GGUF binary extraction, database persistence, group validation, and multi-agent loops:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📄 License

MIT License. Designed and built for seamless pairing with Herdr and local open-source LLMs.
