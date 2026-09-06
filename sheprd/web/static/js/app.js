/**
 * Sheprd Frontend Controller
 * Digital Green UI, real-time agent lifecycle, GGUF inspection, Herdr spawning.
 */

let state = {
  agents: [],
  systemStatus: null,
  activeChatAgent: null,
  chatHistories: {},
  activeLogAgent: null,
  logInterval: null,
};

// Preset Personas for quick setup
const PERSONA_PRESETS = {
  architect: {
    identity: "Senior Python Architect & Code Reviewer",
    personality: "Precise, pragmatic, uncompromising on security, speaks concisely.",
    job: "Analyze codebases, detect vulnerabilities, review pull requests, and optimize algorithms.",
  },
  sysadmin: {
    identity: "Hardened Linux Systems Administrator",
    personality: "Succinct, dry humor, terminal-native, highly cautious with root permissions.",
    job: "Diagnose Linux errors, write bash scripts, manage systemd services, and troubleshoot networking.",
  },
  researcher: {
    identity: "Deep Technical Research Specialist",
    personality: "Thorough, evidence-based, provides structured citations and executive summaries.",
    job: "Synthesize complex technical concepts, compare architectures, and provide step-by-step guides.",
  },
  assistant: {
    identity: "General Digital Green Shepherd Assistant",
    personality: "Helpful, alert, articulate, and proactive.",
    job: "Assist with day-to-day coding, terminal commands, and general queries.",
  }
};

// CSRF Token Helper (S2)
function getCsrfToken() {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute('content') : '';
}

// Intercept fetch to automatically include X-Sheprd-Token on mutating requests (S2)
const originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  if (method !== 'GET') {
    options.headers = options.headers || {};
    if (options.headers instanceof Headers) {
      options.headers.set('X-Sheprd-Token', getCsrfToken());
    } else {
      options.headers['X-Sheprd-Token'] = getCsrfToken();
    }
  }
  return originalFetch(url, options);
};

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupInspectButton();
  setupCreateForm();
  setupChat();
  setupPresets();
  setupMCPModal();
  loadInitialData();
  setupSSE();
});

// Toast notification helper
function showToast(message, type = "normal") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Navigation Tabs
function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      if (targetId === "logs-tab") {
        startLogStreaming();
      } else {
        stopLogStreaming();
      }

      if (targetId === "groups-tab") {
        renderGroups();
      }

      if (targetId === "tools-tab") {
        loadToolsAndMCPServers();
      }
    });
  });
}

function switchTab(tabId) {
  const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (btn) btn.click();
}

// Initial Data Fetching
async function loadInitialData() {
  await fetchSystemStatus();
  await fetchAgents();
  await loadToolsAndMCPServers();
}

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    state.systemStatus = data;

    document.getElementById("stat-cpu").textContent = `${data.cpu_model.split(" ")[0]} (${data.logical_cores}T)`;
    document.getElementById("stat-ram").textContent = `${data.available_ram_gb}G free / ${data.total_ram_gb}G`;
    
    const gpuText = data.gpu_devices && data.gpu_devices.length > 0
      ? data.gpu_devices[0].split("/")[0].replace("Advanced Micro Devices, Inc. [AMD/ATI]", "AMD").trim()
      : "CPU Only";
    document.getElementById("stat-gpu").textContent = `${gpuText} ${data.vulkan_supported ? '[VK]' : ''}`;
    
    const herdrEl = document.getElementById("stat-herdr");
    herdrEl.textContent = data.herdr_available ? "ONLINE" : "OFFLINE";
    herdrEl.style.color = data.herdr_available ? "var(--accent-green)" : "var(--accent-red)";
  } catch (err) {
    console.error("Status fetch error:", err);
  }
}

async function fetchAgents() {
  try {
    const res = await fetch("/api/agents");
    if (!res.ok) return;
    const agents = await res.json();
    state.agents = agents;
    renderAgents(agents);
    updateChatSidebar(agents);
    updateLogsDropdown(agents);
  } catch (err) {
    console.error("Agents fetch error:", err);
  }
}

// Real-time SSE
function setupSSE() {
  try {
    const es = new EventSource("/api/events");
    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "agents_update") {
          state.agents = payload.agents;
          renderAgents(payload.agents);
          updateChatSidebar(payload.agents);
        }
      } catch (e) {}
    };
  } catch (e) {
    console.warn("SSE not supported, falling back to polling");
    setInterval(fetchAgents, 5000);
  }
}

// Render Agent Grid
function renderAgents(agents) {
  const container = document.getElementById("agents-grid");
  if (!container) return;

  if (agents.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 48px; border: 1px dashed var(--border-green); border-radius: 6px;">
        <h3 style="color: var(--accent-green); margin-bottom: 12px;">No agents deployed yet</h3>
        <p style="color: var(--text-dim); margin-bottom: 18px;">Deploy a llama.cpp model to spin up your first digital green agent.</p>
        <button class="btn btn-primary" onclick="switchTab('deploy-tab')">+ Deploy New Agent</button>
      </div>
    `;
    return;
  }

  container.innerHTML = agents.map(agent => {
    const isRunning = agent.status === "running";
    const statusClass = isRunning ? "running" : (agent.status === "error" ? "error" : "stopped");
    const statusText = isRunning ? `ONLINE (:${agent.port})` : agent.status.toUpperCase();
    const modelBase = agent.model_path.split("/").pop();

    const groupsHtml = agent.groups.map(g => `<span class="meta-pill">grp: <strong>${escapeHtml(g)}</strong></span>`).join(" ");

    return `
      <div class="agent-card" id="card-${agent.name}">
        <div class="agent-card-header">
          <div>
            <div class="agent-name">${escapeHtml(agent.name)}</div>
            <div class="agent-identity">${escapeHtml(agent.identity)}</div>
          </div>
          <span class="status-badge ${statusClass}">● ${statusText}</span>
        </div>

        <div class="agent-personality"><strong>Personality:</strong> ${escapeHtml(agent.personality)}</div>
        <div class="agent-job"><strong>Job:</strong> ${escapeHtml(agent.job)}</div>

        <div class="card-meta-row">
          <span class="meta-pill">model: <strong>${escapeHtml(modelBase)}</strong></span>
          <span class="meta-pill">ctx: <strong>${agent.context_size}</strong></span>
          <span class="meta-pill">gpu-layers: <strong>${agent.n_gpu_layers}</strong></span>
          <span class="meta-pill ${agent.callable_by_agents ? 'highlight' : ''}">
            callable: <strong>${agent.callable_by_agents ? 'YES' : 'NO'}</strong>
          </span>
          ${agent.telegram_enabled ? `<span class="meta-pill highlight">✈ TG: <strong>ACTIVE</strong></span>` : ''}
          ${agent.tools && agent.tools.length > 0 ? `<span class="meta-pill">tools: <strong>${agent.tools.length}</strong></span>` : ''}
          ${groupsHtml}
        </div>

        <div class="agent-card-actions">
          ${isRunning 
            ? `<button class="btn btn-danger" onclick="stopAgent('${agent.name}')">⏹ Stop</button>`
            : `<button class="btn btn-primary" onclick="startAgent('${agent.name}')">▶ Start</button>`
          }
          <button class="btn btn-herdr" onclick="spawnInHerdr('${agent.name}')" title="Spawn interactive terminal session in Herdr">
            ⚡ Spawn in Herdr
          </button>
          <button class="btn btn-secondary" onclick="openAgentChat('${agent.name}')">
            💬 Chat
          </button>
          <button class="btn btn-secondary" onclick="viewAgentLogs('${agent.name}')">
            📜 Logs
          </button>
          <button class="btn btn-danger" style="margin-left: auto;" onclick="deleteAgent('${agent.name}')" title="Delete agent">
            🗑
          </button>
        </div>
      </div>
    `;
  }).join("");
}

// Preset button handlers
function setupPresets() {
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      const p = PERSONA_PRESETS[presetKey];
      if (!p) return;

      document.getElementById("form-identity").value = p.identity;
      document.getElementById("form-personality").value = p.personality;
      document.getElementById("form-job").value = p.job;
      showToast(`Applied '${presetKey}' persona preset.`);
    });
  });
}

// Model Auto-Inspection
function setupInspectButton() {
  const btn = document.getElementById("btn-inspect-model");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const pathInput = document.getElementById("form-model-path");
    const modelPath = pathInput.value.trim();
    if (!modelPath) {
      showToast("Please enter a model path first.", "normal");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Inspecting...";

    try {
      const res = await fetch("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath }),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        showToast(data.error || "Inspection failed.", "normal");
        return;
      }

      const meta = data.model_metadata;
      const rec = data.recommended_config;

      // Populate form fields automatically
      document.getElementById("form-context-size").value = rec.context_size;
      document.getElementById("form-gpu-layers").value = rec.n_gpu_layers;
      document.getElementById("form-threads").value = rec.threads;
      document.getElementById("form-port").value = rec.port;
      document.getElementById("form-template").value = rec.template_kind;

      // If agent name is empty, suggest one
      const nameInput = document.getElementById("form-name");
      if (!nameInput.value) {
        const cleanName = meta.model_name
          .toLowerCase()
          .replace(/[^a-z0-9_-]/g, "-")
          .substring(0, 24);
        nameInput.value = cleanName;
      }

      // Display inspection box
      const box = document.getElementById("inspection-result-box");
      box.classList.add("visible");
      document.getElementById("inspect-arch").textContent = meta.architecture;
      document.getElementById("inspect-quant").textContent = meta.quantization;
      document.getElementById("inspect-layers").textContent = `${meta.layer_count} layers`;
      document.getElementById("inspect-ctx").textContent = `${meta.context_length} native`;
      document.getElementById("inspect-size").textContent = `${meta.file_size_gb} GB`;
      document.getElementById("inspect-rec-gpu").textContent = `${rec.n_gpu_layers} layers`;

      showToast(`Model verified: ${meta.architecture} (${meta.quantization}). Settings auto-configured!`);
    } catch (err) {
      showToast(`Inspection error: ${err}`, "normal");
    } finally {
      btn.disabled = false;
      btn.textContent = "🔍 Auto-Inspect Model";
    }
  });
}

// Agent Form Creation
function setupCreateForm() {
  const form = document.getElementById("create-agent-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("form-name").value.trim();
    const identity = document.getElementById("form-identity").value.trim();
    const personality = document.getElementById("form-personality").value.trim();
    const job = document.getElementById("form-job").value.trim();
    const modelPath = document.getElementById("form-model-path").value.trim();

    const port = parseInt(document.getElementById("form-port").value, 10);
    const contextSize = parseInt(document.getElementById("form-context-size").value, 10);
    const nGpuLayers = parseInt(document.getElementById("form-gpu-layers").value, 10);
    const threads = parseInt(document.getElementById("form-threads").value, 10);
    const templateKind = document.getElementById("form-template").value;

    const tgEnabled = document.getElementById("form-tg-enabled").checked;
    const tgToken = document.getElementById("form-tg-token").value.trim();

    const callableBy = document.getElementById("form-callable").checked;
    const groupsRaw = document.getElementById("form-groups").value.trim();
    const groups = groupsRaw ? groupsRaw.split(",").map(g => g.trim()).filter(Boolean) : ["default"];

    const toolCheckboxes = document.querySelectorAll('input[name="form_tool"]:checked');
    const tools = Array.from(toolCheckboxes).map(cb => cb.value);

    const payload = {
      name,
      identity,
      personality,
      job,
      model_path: modelPath,
      port,
      context_size: contextSize,
      n_gpu_layers: nGpuLayers,
      threads,
      template_kind: templateKind,
      telegram_enabled: tgEnabled,
      telegram_bot_token: tgToken,
      callable_by_agents: callableBy,
      groups,
      tools,
    };

    try {
      const res = await fetch("/api/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        showToast(data.error || "Failed to create agent.", "normal");
        return;
      }

      showToast(`Agent '${name}' created! Herdr launcher script installed at ~/.local/bin/${name}`);
      form.reset();
      document.getElementById("inspection-result-box").classList.remove("visible");
      await fetchAgents();
      switchTab("agents-tab");
    } catch (err) {
      showToast(`Network error: ${err}`);
    }
  });
}

// Agent Actions
async function startAgent(name) {
  showToast(`Starting ${name}...`);
  try {
    const res = await fetch(`/api/agents/${name}/start`, { method: "POST" });
    const data = await res.json();
    if (res.ok && data.ok) {
      showToast(`Agent '${name}' is now ONLINE!`);
    } else {
      showToast(`Start failed: ${data.error}`);
    }
    await fetchAgents();
  } catch (err) {
    showToast(`Error starting agent: ${err}`);
  }
}

async function stopAgent(name) {
  showToast(`Stopping ${name}...`);
  try {
    const res = await fetch(`/api/agents/${name}/stop`, { method: "POST" });
    const data = await res.json();
    showToast(data.message || `Agent '${name}' stopped.`);
    await fetchAgents();
  } catch (err) {
    showToast(`Error stopping agent: ${err}`);
  }
}

async function spawnInHerdr(name) {
  showToast(`Spawning '${name}' in Herdr...`);
  try {
    const res = await fetch(`/api/agents/${name}/spawn-herdr`, { method: "POST" });
    const data = await res.json();
    if (res.ok && data.ok) {
      showToast(`Spawned in Herdr! Switch to your Herdr terminal to interact with ${name}.`);
    } else {
      showToast(`Herdr spawn: ${data.message || data.error}`);
    }
    await fetchAgents();
  } catch (err) {
    showToast(`Failed to trigger Herdr: ${err}`);
  }
}

async function deleteAgent(name) {
  if (!confirm(`Are you sure you want to delete agent '${name}'? This removes its server and launcher.`)) {
    return;
  }
  try {
    const res = await fetch(`/api/agents/${name}`, { method: "DELETE" });
    const data = await res.json();
    showToast(data.message);
    await fetchAgents();
  } catch (err) {
    showToast(`Error deleting agent: ${err}`);
  }
}

// Chat Console
function setupChat() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");

  const sendMessage = async () => {
    const text = input.value.trim();
    if (!text || !state.activeChatAgent) return;

    input.value = "";
    appendChatMessage(state.activeChatAgent, "user", text);

    sendBtn.disabled = true;
    input.disabled = true;

    try {
      const history = state.chatHistories[state.activeChatAgent] || [];
      const res = await fetch(`/api/agents/${state.activeChatAgent}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, history }),
      });

      const data = await res.json();
      if (res.ok && data.ok) {
        appendChatMessage(state.activeChatAgent, "assistant", data.response, data.tool_calls);
      } else {
        appendChatMessage(state.activeChatAgent, "system", `Error: ${data.error}`);
      }
    } catch (err) {
      appendChatMessage(state.activeChatAgent, "system", `Network error: ${err}`);
    } finally {
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  };

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

function updateChatSidebar(agents) {
  const container = document.getElementById("chat-agent-list");
  if (!container) return;

  container.innerHTML = agents.map(a => `
    <div class="chat-agent-item ${a.name === state.activeChatAgent ? 'active' : ''}" onclick="selectChatAgent('${a.name}')">
      <div>
        <div style="font-weight: 700; color: var(--accent-green);">${escapeHtml(a.name)}</div>
        <div style="font-size: 11px; color: var(--text-dim);">${escapeHtml(a.identity)}</div>
      </div>
      <span style="font-size: 10px; color: ${a.status === 'running' ? 'var(--accent-green)' : 'var(--text-dim)'}">
        ● ${a.status.toUpperCase()}
      </span>
    </div>
  `).join("");

  if (!state.activeChatAgent && agents.length > 0) {
    selectChatAgent(agents[0].name);
  }
}

function selectChatAgent(name) {
  state.activeChatAgent = name;
  document.getElementById("chat-active-title").textContent = `Chatting with: ${name}`;
  updateChatSidebar(state.agents);
  renderChatMessages(name);
}

function openAgentChat(name) {
  selectChatAgent(name);
  switchTab("chat-tab");
}

function renderChatMessages(name) {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  const history = state.chatHistories[name] || [];

  container.innerHTML = history.map(msg => {
    let toolBadgeHtml = "";
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      const toolNames = msg.tool_calls.map(tc => escapeHtml(tc.name || "tool")).join(", ");
      toolBadgeHtml = `<div class="tool-call-badge">⚡ Executed Tools: <strong>${toolNames}</strong></div>`;
    }
    return `
      <div class="chat-bubble ${msg.role}">
        <div class="sender-label">${msg.role === 'user' ? 'YOU' : name.toUpperCase()}</div>
        ${toolBadgeHtml}
        <div>${escapeHtml(msg.content)}</div>
      </div>
    `;
  }).join("");

  container.scrollTop = container.scrollHeight;
}

function appendChatMessage(agentName, role, content, toolCalls = []) {
  if (!state.chatHistories[agentName]) state.chatHistories[agentName] = [];
  state.chatHistories[agentName].push({ role, content, tool_calls: toolCalls });
  if (state.activeChatAgent === agentName) {
    renderChatMessages(agentName);
  }
}

// Groups & Swarms View
async function renderGroups() {
  const container = document.getElementById("groups-container");
  if (!container) return;

  try {
    const res = await fetch("/api/groups");
    const data = await res.json();
    const groups = data.groups || {};

    if (Object.keys(groups).length === 0) {
      container.innerHTML = `<div style="color: var(--text-dim);">No agent groups configured yet.</div>`;
      return;
    }

    container.innerHTML = Object.entries(groups).map(([gName, members]) => `
      <div class="form-panel" style="margin-bottom: 24px; max-width: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <h3 style="color: var(--accent-green);">Group: [ ${escapeHtml(gName)} ] (${members.length} agents)</h3>
          <button class="btn btn-secondary" onclick="promptGroupBroadcast('${gName}')">📢 Broadcast Task</button>
        </div>
        <div class="card-meta-row">
          ${members.map(m => `
            <div class="inspect-stat-card" style="min-width: 200px;">
              <div class="label">${escapeHtml(m.identity)}</div>
              <div class="value">${escapeHtml(m.name)}</div>
              <div style="font-size: 11px; margin-top: 4px; color: ${m.status === 'running' ? 'var(--accent-green)' : 'var(--text-dim)'}">
                ● ${m.status.toUpperCase()} | Port ${m.port}
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    `).join("");
  } catch (err) {
    console.error("Groups fetch error:", err);
  }
}

async function promptGroupBroadcast(groupName) {
  const prompt = window.prompt(`Enter task or query to broadcast to all agents in group '${groupName}':`);
  if (!prompt) return;

  showToast(`Broadcasting task to group '${groupName}'...`);
  try {
    const res = await fetch(`/api/groups/${groupName}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, caller: "Web-Console" }),
    });
    const data = await res.json();
    const results = data.results || [];
    
    let summary = `Broadcast Results for [${groupName}]:\n\n`;
    results.forEach(r => {
      summary += `--- Agent ${r.target_agent} ---\n${r.response || r.error}\n\n`;
    });
    alert(summary);
  } catch (err) {
    showToast(`Broadcast failed: ${err}`);
  }
}

// System Logs Streaming
function updateLogsDropdown(agents) {
  const select = document.getElementById("log-agent-select");
  if (!select) return;
  const currentVal = select.value;
  select.innerHTML = agents.map(a => `<option value="${a.name}">${a.name} (Port ${a.port})</option>`).join("");
  if (currentVal && agents.some(a => a.name === currentVal)) {
    select.value = currentVal;
  } else if (agents.length > 0) {
    select.value = agents[0].name;
  }
}

function viewAgentLogs(name) {
  const select = document.getElementById("log-agent-select");
  if (select) select.value = name;
  switchTab("logs-tab");
  fetchLogs();
}

async function fetchLogs() {
  const select = document.getElementById("log-agent-select");
  const consoleEl = document.getElementById("logs-console");
  if (!select || !consoleEl) return;
  const agentName = select.value;
  if (!agentName) return;

  try {
    const res = await fetch(`/api/agents/${agentName}/logs`);
    const data = await res.json();
    consoleEl.textContent = data.logs || "No logs available.";
    consoleEl.scrollTop = consoleEl.scrollHeight;
  } catch (err) {
    consoleEl.textContent = `Error reading logs: ${err}`;
  }
}

function startLogStreaming() {
  fetchLogs();
  if (!state.logInterval) {
    state.logInterval = setInterval(fetchLogs, 2500);
  }
}

function stopLogStreaming() {
  if (state.logInterval) {
    clearInterval(state.logInterval);
    state.logInterval = null;
  }
}

// Utility
function escapeHtml(text) {
  if (!text) return "";
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}

// ---------------------------------------------------------------------------
// Tools & MCP Skills Hub Controller
// ---------------------------------------------------------------------------

const MCP_PRESETS = {
  custom: { name: "", command: "", args: "", desc: "" },
  duckduckgo: {
    name: "duckduckgo",
    command: "npx",
    args: "-y @modelcontextprotocol/server-duckduckgo",
    desc: "Real-time web search capabilities via DuckDuckGo",
  },
  brave: {
    name: "brave-search",
    command: "npx",
    args: "-y @modelcontextprotocol/server-brave-search",
    desc: "Brave Web Search API for live web queries",
  },
  sqlite: {
    name: "sqlite",
    command: "uvx",
    args: "mcp-server-sqlite --db-path ./sheprd.db",
    desc: "Direct database queries and schema inspection on SQLite",
  },
  filesystem: {
    name: "filesystem",
    command: "npx",
    args: "-y @modelcontextprotocol/server-filesystem /home/sachsen",
    desc: "Secure local filesystem access for reading local files",
  },
  memory: {
    name: "memory",
    command: "npx",
    args: "-y @modelcontextprotocol/server-memory",
    desc: "Persistent knowledge graph and agent memory graph",
  },
};

function applyMCPPreset(key) {
  const p = MCP_PRESETS[key];
  if (!p) return;
  document.getElementById("mcp-name").value = p.name;
  document.getElementById("mcp-command").value = p.command;
  document.getElementById("mcp-args").value = p.args;
  document.getElementById("mcp-desc").value = p.desc;
}

function openAddMCPModal() {
  const modal = document.getElementById("add-mcp-modal");
  if (modal) modal.style.display = "flex";
}

function closeAddMCPModal() {
  const modal = document.getElementById("add-mcp-modal");
  if (modal) modal.style.display = "none";
}

function setupMCPModal() {
  const form = document.getElementById("add-mcp-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("mcp-name").value.trim();
    const command = document.getElementById("mcp-command").value.trim();
    const rawArgs = document.getElementById("mcp-args").value.trim();
    const description = document.getElementById("mcp-desc").value.trim();

    if (!name || !command) {
      showToast("Server identifier and command binary are required.", "normal");
      return;
    }

    const args = rawArgs ? rawArgs.split(/\s+/).filter(Boolean) : [];

    try {
      const res = await fetch("/api/mcp/servers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, command, args, description, enabled: true }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        showToast(data.error || "Failed to add MCP server.", "normal");
        return;
      }

      showToast(`MCP server '${name}' connected successfully!`);
      form.reset();
      closeAddMCPModal();
      await loadToolsAndMCPServers();
    } catch (err) {
      showToast(`Error connecting MCP server: ${err}`, "normal");
    }
  });
}

async function loadToolsAndMCPServers() {
  try {
    const [toolsRes, serversRes] = await Promise.all([
      fetch("/api/tools"),
      fetch("/api/mcp/servers"),
    ]);

    if (toolsRes.ok) {
      const data = await toolsRes.json();
      renderCoreTools(data.tools || []);
    }

    if (serversRes.ok) {
      const data = await serversRes.json();
      renderMCPServers(data.servers || []);
    }
  } catch (err) {
    console.error("Error loading tools / MCP servers:", err);
  }
}

function renderCoreTools(tools) {
  const container = document.getElementById("core-tools-list");
  if (!container) return;

  const coreTools = tools.filter(t => t.type === "core");
  if (coreTools.length === 0) {
    container.innerHTML = `<div style="color: var(--text-dim); padding: 12px;">No core tools registered.</div>`;
    return;
  }

  container.innerHTML = coreTools.map(t => `
    <div class="tool-item-card">
      <div class="tool-header-line">
        <div class="tool-name"><code>${escapeHtml(t.name)}</code></div>
        <span class="tool-badge core">Built-in</span>
      </div>
      <div class="tool-desc">${escapeHtml(t.description)}</div>
    </div>
  `).join("");
}

function renderMCPServers(servers) {
  const container = document.getElementById("mcp-servers-list");
  const countBadge = document.getElementById("mcp-count-badge");
  if (countBadge) {
    countBadge.textContent = `${servers.length} Server${servers.length === 1 ? '' : 's'}`;
  }
  if (!container) return;

  if (servers.length === 0) {
    container.innerHTML = `
      <div style="color: var(--text-dim); padding: 24px; text-align: center; border: 1px dashed var(--border-green); border-radius: 4px;">
        <div style="margin-bottom: 8px; font-weight: 700; color: var(--accent-green);">No external MCP servers connected</div>
        <div style="font-size: 12px; margin-bottom: 14px;">Hook into DuckDuckGo, Brave Search, SQLite, Filesystem, or any stdio MCP server.</div>
        <button class="btn btn-primary" onclick="openAddMCPModal()">+ Connect First MCP Server</button>
      </div>
    `;
    return;
  }

  container.innerHTML = servers.map(s => {
    const cmdStr = `${s.command} ${(s.args || []).join(" ")}`.trim();
    return `
      <div class="tool-item-card" id="mcp-card-${escapeHtml(s.name)}">
        <div class="tool-header-line">
          <div>
            <div class="tool-name" style="color: var(--accent-amber);">${escapeHtml(s.name)}</div>
            <div style="font-size: 11px; color: var(--text-dim); margin-top: 2px;"><code>${escapeHtml(cmdStr)}</code></div>
          </div>
          <div style="display: flex; gap: 6px; align-items: center;">
            <span class="tool-badge mcp">${s.enabled ? 'Enabled' : 'Disabled'}</span>
            <button class="btn btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="testMCPServer('${escapeHtml(s.name)}')">⚡ Test</button>
            <button class="btn btn-danger" style="padding: 2px 8px; font-size: 11px;" onclick="deleteMCPServer('${escapeHtml(s.name)}')">🗑</button>
          </div>
        </div>
        <div class="tool-desc">${escapeHtml(s.description || "No description provided.")}</div>
      </div>
    `;
  }).join("");
}

async function deleteMCPServer(name) {
  if (!confirm(`Are you sure you want to disconnect MCP server '${name}'?`)) return;
  try {
    const res = await fetch(`/api/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok && data.ok) {
      showToast(`MCP server '${name}' disconnected.`);
      await loadToolsAndMCPServers();
    } else {
      showToast(data.error || "Failed to delete server.", "normal");
    }
  } catch (err) {
    showToast(`Error deleting MCP server: ${err}`, "normal");
  }
}

async function testMCPServer(name) {
  showToast(`Testing MCP server '${name}'...`);
  try {
    const res = await fetch(`/api/mcp/servers/${encodeURIComponent(name)}/test`, { method: "POST" });
    const data = await res.json();
    if (res.ok && data.ok) {
      showToast(`MCP '${name}' OK! Discovered ${data.tools_count} tools: ${data.tools.join(', ')}`);
      await loadToolsAndMCPServers();
    } else {
      showToast(`Test failed: ${data.error || 'Server did not respond'}`, "normal");
    }
  } catch (err) {
    showToast(`Error testing MCP server: ${err}`, "normal");
  }
}

// Global exports for inline HTML event handlers
window.applyMCPPreset = applyMCPPreset;
window.openAddMCPModal = openAddMCPModal;
window.closeAddMCPModal = closeAddMCPModal;
window.deleteMCPServer = deleteMCPServer;
window.testMCPServer = testMCPServer;

