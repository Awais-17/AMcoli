const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');
const https = require('https');
const { exec } = require('child_process');
const readline = require('readline');

// Colored Output Helpers
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  red: '\x1b[31m',
  gray: '\x1b[90m'
};

// Default Configuration (overridable via flags / environment variables)
const CONFIG = {
  apiUrl: process.env.AMCOLI_API_URL || 'http://127.0.0.1:11434/v1', // Defaults to local Ollama
  model: process.env.AMCOLI_MODEL || 'qwen-coder-32b',               // Defaults to Qwen Coder
  apiKey: process.env.AMCOLI_API_KEY || '',                          // Bearer token for remote endpoints
  cwd: process.cwd(),                                                // Working directory for tools
  autoApprove: false,                                                // --yes: approve every tool call
  autoReadonly: false,                                               // --auto: auto-approve read-only tools
  stream: false,                                                     // --stream: stream responses
  maxContext: 24,                                                    // Max messages kept in context
  maxOutputChars: 50000,                                             // Tool output truncation cap
  timeoutMs: 120000,                                                 // Command / HTTP timeout
  historyFile: path.join(os.homedir(), '.amcoli', 'agent-history.jsonl')
};

const BANNER_LOGO = [
  '\x1b[1;31m ▄▄▄▄▄▄▄ ▄▄   ▄▄ \x1b[1;37m▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ \x1b[0m',
  '\x1b[1;31m█       █  █▄█  █\x1b[1;37m       █       █  █ █  █       █\x1b[0m',
  '\x1b[1;31m█   ▄   █   █   █\x1b[1;37m       █   ▄   █  █▄█  █   ▄   █\x1b[0m',
  '\x1b[1;31m█  █▄█  █       █\x1b[1;37m     ▄▄█  █ █  █       █  █ █  █\x1b[0m',
  '\x1b[1;31m█       █       █\x1b[1;37m    █  █  █▄█  █       █  █▄█  █\x1b[0m',
  '\x1b[1;31m█   ▄   █ ██▄██ █\x1b[1;37m    █▄▄█       █   ▄   █       █\x1b[0m',
  '\x1b[1;31m█▄▄█ █▄▄█▄█   █▄█\x1b[1;37m▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄█ █▄▄█▄▄▄▄▄▄▄█\x1b[0m'
];

// Tools that only read the filesystem / system state (never mutate or execute)
const READONLY_TOOLS = new Set(['list_dir', 'read_file', 'search_files', 'get_system_info']);

// Lazily-created readline interface (only created when the interactive loop starts)
let rl = null;

function ensureRL() {
  if (!rl) {
    rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    // Exit gracefully when stdin closes (piped input, Ctrl+D, or EOF)
    rl.on('close', () => {
      console.log(`\n${colors.yellow}Exiting AMcoli Agentic CLI. Bye!${colors.reset}\n`);
      process.exit(0);
    });
  }
  return rl;
}

/**
 * Build the system prompt describing tools, rules, and the local environment.
 */
function buildSystemPrompt() {
  const cpus = os.cpus();
  const totalMem = (os.totalmem() / (1024 ** 3)).toFixed(1);
  const freeMem = (os.freemem() / (1024 ** 3)).toFixed(1);
  let diskFree = 'n/a';
  try {
    const s = fs.statfsSync(CONFIG.cwd);
    diskFree = `${(s.bavail * s.bsize / (1024 ** 3)).toFixed(1)} GB free`;
  } catch (_) { /* statfs unavailable */ }

  return `You are "AMcoli Agent", a powerful, autonomous coding assistant.
You help the user develop, debug, and understand their code, with a focus on C++ and systems programming (this project is a Mixture-of-Experts disk-streaming inference engine).
You run locally in a terminal on the user's computer.

## Local Environment (injected at startup)
- Platform : ${os.platform()} ${os.release()} (${os.arch()})
- Working dir: ${CONFIG.cwd}
- CPU cores: ${cpus.length} (${cpus[0] ? cpus[0].model.trim() : 'unknown'})
- RAM: ${freeMem} GB free / ${totalMem} GB
- Disk: ${diskFree}
- Default model: ${CONFIG.model}

## How to use tools
To interact with the local system you invoke tools by responding with a JSON code block.
Write the JSON block exactly like this (no other JSON keys, no text inside the same block):
\`\`\`json
{
  "tool": "tool_name",
  "args": { "arg1": "value" }
}
\`\`\`

You may call ONLY ONE tool per turn. After a tool call the system executes it and returns the result in the next message. Never make up tool results; always wait for the system.

## Supported Tools

1. \`list_dir\`: List the contents of a directory.
   - Args: {"path": "string"} (use "." for current directory)

2. \`read_file\`: Read a text file (optional \`start\` line, 1-based, and \`limit\` max lines).
   - Args: {"path": "string", "start": "number (optional)", "limit": "number (optional)"}

3. \`search_files\`: Case-sensitive substring/regex search across a directory tree (skips node_modules, .git, build, .models, etc.).
   - Args: {"pattern": "string", "path": "string (optional, default cwd)", "regex": "boolean (optional)"}

4. \`get_system_info\`: Show CPU/RAM/disk/OS facts. No args required.

5. \`write_file\`: Create or overwrite a file.
   - Args: {"path": "string", "content": "string"}

6. \`edit_file\`: Apply a single find-and-replace edit to an existing file.
   - Args: {"path": "string", "old": "exact text to find", "new": "replacement text", "all": "boolean (optional, replace every occurrence)"}

7. \`run_command\`: Run a shell command (compiling, tests, git, etc.) in the working directory. Commands time out after ${CONFIG.timeoutMs / 1000}s and long output is truncated.
   - Args: {"command": "string"}

8. \`run_amcoli\`: Run the local AMcoli binary (e.g. \`list\`, \`check\`, \`info <alias>\`, \`version\`, \`bench\`, \`recommend\`, \`pull <alias> --dry-run\`).
   - Args: {"args": ["list"]} — array of command-line arguments

## Core Instructions
- Analyze the user's request. If you need to view files, search the codebase, or build/run tests to answer, immediately call the appropriate tool.
- Explain your reasoning briefly before calling a tool.
- Keep your changes precise and minimal.
- When finished, summarize your work and output a regular text response.`;
}

// Conversation State
const chatHistory = [{ role: 'system', content: buildSystemPrompt() }];

/* ── History persistence (JSONL in ~/.amcoli/agent-history.jsonl) ─────── */

function ensureHistoryDir() {
  try { fs.mkdirSync(path.dirname(CONFIG.historyFile), { recursive: true }); }
  catch (_) { /* ignore */ }
}

function appendHistory(message) {
  try {
    ensureHistoryDir();
    fs.appendFileSync(CONFIG.historyFile, JSON.stringify(message) + '\n', 'utf8');
  } catch (_) { /* persistence is best-effort */ }
}

function loadHistory() {
  try {
    if (!fs.existsSync(CONFIG.historyFile)) return;
    const lines = fs.readFileSync(CONFIG.historyFile, 'utf8').trim().split('\n');
    for (const line of lines) {
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg && msg.role && msg.content) chatHistory.push(msg);
      } catch (_) { /* skip corrupt lines */ }
    }
    trimHistory();
  } catch (_) { /* ignore */ }
}

function clearHistoryFile() {
  try { fs.unlinkSync(CONFIG.historyFile); } catch (_) { /* ignore */ }
}

/**
 * Keep context bounded: always retain the system message, drop the oldest
 * user/assistant pairs once the limit is exceeded.
 */
function trimHistory() {
  while (chatHistory.length > CONFIG.maxContext) {
    const idx = chatHistory.findIndex((m, i) => i > 0 && m.role === 'user');
    if (idx === -1) break;
    chatHistory.splice(idx, 2);
  }
}

/* ── LLM HTTP client (OpenAI-compatible, HTTPS-aware, optional streaming) ─ */

function queryLLM(messages) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(CONFIG.apiUrl);
    const isHttps = parsedUrl.protocol === 'https:';
    const transport = isHttps ? https : http;

    const postData = JSON.stringify({
      model: CONFIG.model,
      messages,
      temperature: 0.2,
      stream: CONFIG.stream
    });

    const headers = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(postData)
    };
    if (CONFIG.apiKey) headers.Authorization = `Bearer ${CONFIG.apiKey}`;

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: `${parsedUrl.pathname}/chat/completions`.replace(/\/\//g, '/'),
      method: 'POST',
      headers
    };

    const req = transport.request(options, (res) => {
      let raw = '';
      let streamed = '';
      let streamingOk = false;

      const handleStreamLine = (line) => {
        if (!line.startsWith('data:')) return;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') { streamingOk = true; return; }
        try {
          const json = JSON.parse(payload);
          const delta = json.choices && json.choices[0] && json.choices[0].delta;
          if (delta && typeof delta.content === 'string') {
            streamed += delta.content;
            process.stdout.write(delta.content);
          }
        } catch (_) { /* partial/keep-alive line */ }
      };

      res.on('data', (chunk) => {
        raw += chunk;
        let idx;
        while ((idx = raw.indexOf('\n')) !== -1) {
          const line = raw.slice(0, idx).trim();
          raw = raw.slice(idx + 1);
          if (line) handleStreamLine(line);
        }
      });

      res.on('end', () => {
        if (streamingOk && streamed) {
          process.stdout.write('\n');
          resolve(streamed);
          return;
        }
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            const json = JSON.parse(raw);
            resolve(json.choices[0].message.content);
          } catch (e) {
            reject(new Error(`Failed to parse API response: ${e.message}. Response was: ${raw.slice(0, 500)}`));
          }
        } else {
          reject(new Error(`API returned status ${res.statusCode}: ${raw.slice(0, 500)}`));
        }
      });
    });

    req.setTimeout(CONFIG.timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${CONFIG.timeoutMs / 1000}s`));
    });
    req.on('error', (e) => reject(e));
    req.write(postData);
    req.end();
  });
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

function truncate(text, max) {
  if (typeof text !== 'string' || text.length <= max) return text || '';
  return text.slice(0, max) + `\n...[output truncated at ${max} characters]`;
}

/* ── Tool Implementations ─────────────────────────────────────────────── */

const SKIP_DIRS = new Set(['node_modules', '.git', 'build', '.models', 'dist', 'Testing', 'bin', 'lib', '.cache', '.venv', '__pycache__']);
const MAX_SEARCH_FILES = 600;
const MAX_SEARCH_RESULTS = 200;

function walk(dir, depth, onFile, onError) {
  if (depth > 10) return;
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
  catch (err) { if (onError) onError(dir, err); return; }
  for (const e of entries) {
    if (e.isDirectory()) {
      if (!SKIP_DIRS.has(e.name)) walk(path.join(dir, e.name), depth + 1, onFile, onError);
    } else if (e.isFile()) {
      onFile(path.join(dir, e.name));
    }
  }
}

const Tools = {
  list_dir: ({ path: dirPath }) => {
    return new Promise((resolve) => {
      const resolved = path.resolve(CONFIG.cwd, dirPath || '.');
      fs.readdir(resolved, { withFileTypes: true }, (err, files) => {
        if (err) return resolve(`Error listing directory: ${err.message}`);
        const list = files.map((f) => `${f.isDirectory() ? '[DIR]  ' : '[FILE] '} ${f.name}`).join('\n');
        resolve(list || '(Directory is empty)');
      });
    });
  },

  read_file: ({ path: filePath, start, limit }) => {
    return new Promise((resolve) => {
      const resolved = path.resolve(CONFIG.cwd, filePath || '');
      fs.stat(resolved, (serr, stats) => {
        if (serr) return resolve(`Error reading file: ${serr.message}`);
        if (stats.size > 5 * 1024 * 1024) {
          return resolve(`Error: file is ${(stats.size / 1024 / 1024).toFixed(1)} MB; too large to read in full. Use search_files or read with a line range.`);
        }
        fs.readFile(resolved, 'utf8', (err, data) => {
          if (err) return resolve(`Error reading file: ${err.message}`);
          const lines = data.split('\n');
          const startLine = Number.isFinite(Number(start)) && Number(start) > 0 ? Math.floor(Number(start)) : 1;
          const maxLines = Number.isFinite(Number(limit)) && Number(limit) > 0 ? Math.floor(Number(limit)) : Infinity;
          const endLine = Math.min(lines.length, startLine - 1 + maxLines);
          const shown = lines.slice(startLine - 1, endLine);
          const header = `${resolved} (${lines.length} lines total)`;
          let body = shown.map((l, i) => `${startLine + i}: ${l}`).join('\n');
          if (endLine < lines.length) body += `\n... [${lines.length - endLine} more lines]`;
          resolve(`${header}\n${truncate(body, CONFIG.maxOutputChars)}`);
        });
      });
    });
  },

  search_files: ({ pattern, path: searchPath, regex }) => {
    return new Promise((resolve) => {
      if (!pattern) return resolve('Error: pattern is required.');
      const root = path.resolve(CONFIG.cwd, searchPath || '.');
      const matcher = regex ? (() => { try { return new RegExp(pattern); } catch (e) { return null; } })() : null;
      if (regex && !matcher) return resolve(`Error: invalid regex "${pattern}"`);
      const results = [];
      let scanned = 0;
      walk(root, 0, (file) => {
        if (scanned >= MAX_SEARCH_FILES) return;
        scanned++;
        if (results.length >= MAX_SEARCH_RESULTS) return;
        try {
          const content = fs.readFileSync(file, 'utf8');
          const lines = content.split('\n');
          for (let i = 0; i < lines.length; i++) {
            let hit;
            if (regex) {
              matcher.lastIndex = 0;
              hit = matcher.test(lines[i]);
            } else {
              hit = lines[i].toLowerCase().includes(pattern.toLowerCase());
            }
            if (hit) {
              results.push(`${file}:${i + 1}: ${lines[i].trim().slice(0, 300)}`);
              if (results.length >= MAX_SEARCH_RESULTS) break;
            }
          }
        } catch (_) { /* unreadable / binary files skipped */ }
      });
      if (!results.length) return resolve(`No matches for "${pattern}" under ${root} (scanned ${scanned} files).`);
      resolve(`Found ${results.length} match(es) under ${root} (scanned ${scanned} files):\n` + results.join('\n'));
    });
  },

  get_system_info: () => {
    const cpus = os.cpus();
    const load = os.loadavg ? os.loadavg().map((x) => x.toFixed(2)).join(', ') : 'n/a';
    let disk = 'n/a';
    try {
      const s = fs.statfsSync(CONFIG.cwd);
      disk = `free ${(s.bavail * s.bsize / (1024 ** 3)).toFixed(1)} GB / total ${(s.blocks * s.bsize / (1024 ** 3)).toFixed(1)} GB`;
    } catch (_) { /* statfs unavailable */ }
    return Promise.resolve([
      `Host      : ${os.hostname()}`,
      `Platform  : ${os.platform()} ${os.release()} (${os.arch()})`,
      `Uptime    : ${Math.floor(os.uptime() / 3600)}h ${Math.floor((os.uptime() % 3600) / 60)}m`,
      `CPU       : ${cpus.length} cores — ${cpus[0] ? cpus[0].model.trim() : 'unknown'}`,
      `Load avg  : ${load}`,
      `RAM       : ${(os.freemem() / (1024 ** 3)).toFixed(1)} GB free / ${(os.totalmem() / (1024 ** 3)).toFixed(1)} GB`,
      `Disk (cwd): ${disk}`,
      `Working dir: ${CONFIG.cwd}`
    ].join('\n'));
  },

  write_file: ({ path: filePath, content }) => {
    return new Promise((resolve) => {
      if (!filePath) return resolve('Error: path is required.');
      const resolved = path.resolve(CONFIG.cwd, filePath);
      const parentDir = path.dirname(resolved);
      fs.mkdirSync(parentDir, { recursive: true });
      fs.writeFile(resolved, String(content ?? ''), 'utf8', (err) => {
        if (err) return resolve(`Error writing file: ${err.message}`);
        resolve(`Successfully wrote ${Buffer.byteLength(String(content ?? ''), 'utf8')} bytes to ${filePath}`);
      });
    });
  },

  edit_file: ({ path: filePath, old: oldText, new: newText, all }) => {
    return new Promise((resolve) => {
      if (!filePath) return resolve('Error: path is required.');
      if (oldText === undefined || oldText === '') return resolve('Error: "old" text is required.');
      const resolved = path.resolve(CONFIG.cwd, filePath);
      fs.readFile(resolved, 'utf8', (err, data) => {
        if (err) return resolve(`Error reading file: ${err.message}`);
        if (!data.includes(oldText)) return resolve(`Error: the specified text was not found in ${filePath}.`);
        const updated = all ? data.split(oldText).join(String(newText ?? '')) : data.replace(oldText, String(newText ?? ''));
        fs.writeFile(resolved, updated, 'utf8', (werr) => {
          if (werr) return resolve(`Error writing file: ${werr.message}`);
          resolve(`Applied edit to ${filePath} (${all ? 'all' : 'first'} occurrence).`);
        });
      });
    });
  },

  run_command: ({ command }) => {
    return new Promise((resolve) => {
      if (!command) return resolve('Error: command is required.');
      exec(command, {
        cwd: CONFIG.cwd,
        timeout: CONFIG.timeoutMs,
        maxBuffer: 10 * 1024 * 1024,
        windowsHide: true
      }, (err, stdout, stderr) => {
        let output = '';
        if (stdout) output += stdout;
        if (stderr) output += `\n[STDERR]\n${stderr}`;
        if (err) {
          output += `\n[ERROR${err.code !== undefined ? `: Code ${err.code}` : ''}]${err.killed ? ' (killed: timed out)' : ''} ${err.message}`;
        }
        resolve(truncate(output.trim(), CONFIG.maxOutputChars) || '(Command ran successfully with no output)');
      });
    });
  },

  run_amcoli: ({ args }) => {
    return new Promise((resolve) => {
      const binaryName = os.platform() === 'win32' ? 'amcoli.exe' : 'amcoli';
      const binaryPath = path.join(__dirname, '..', 'bin', binaryName);
      if (!fs.existsSync(binaryPath)) {
        return resolve(`Error: AMcoli native binary not found at ${binaryPath}. Re-run "npm install" or build from source.`);
      }
      const argv = Array.isArray(args) ? args.map(String) : [];
      exec(`"${binaryPath}" ${argv.map((a) => (a.includes(' ') ? `"${a}"` : a)).join(' ')}`, {
        cwd: CONFIG.cwd,
        timeout: CONFIG.timeoutMs,
        maxBuffer: 10 * 1024 * 1024,
        windowsHide: true
      }, (err, stdout, stderr) => {
        let output = '';
        if (stdout) output += stdout;
        if (stderr) output += `\n[STDERR]\n${stderr}`;
        if (err) output += `\n[ERROR: Code ${err.code}] ${err.message}`;
        resolve(truncate(output.trim(), CONFIG.maxOutputChars) || '(amcoli produced no output)');
      });
    });
  }
};

/**
 * Parse a tool-call JSON block out of an assistant markdown response.
 * Accepts a fenced ```json block, or a bare {"tool":...} object.
 */
function parseToolCall(content) {
  const fenced = content.match(/```json\s*([\s\S]*?)\s*```/);
  const candidate = fenced ? fenced[1].trim() : content.trim();
  try {
    const json = JSON.parse(candidate);
    if (json && typeof json.tool === 'string' && Tools[json.tool]) {
      return json;
    }
  } catch (_) { /* not a JSON tool call */ }
  return null;
}

/**
 * Prompt the user for approval. Honors --yes and --auto modes.
 */
async function askApproval(toolName, args) {
  const isReadonly = READONLY_TOOLS.has(toolName);
  if (CONFIG.autoApprove || (CONFIG.autoReadonly && isReadonly)) {
    console.log(`${colors.gray}[AUTO-APPROVED ${isReadonly ? 'READ-ONLY' : 'EXEC'} TOOL] ${toolName}${colors.reset}`);
    return true;
  }

  console.log(`\n${colors.yellow}┌─── TOOL CALL APPROVAL REQUIRED ──────────────────────────┐${colors.reset}`);
  console.log(`${colors.yellow}│${colors.reset} Tool   : ${colors.bright}${colors.cyan}${toolName}${colors.reset}`);
  console.log(`${colors.yellow}│${colors.reset} Type   : ${colors.bright}${isReadonly ? 'READ-ONLY' : 'WRITE / EXECUTE'}${colors.reset}`);
  console.log(`${colors.yellow}│${colors.reset} Args   : ${JSON.stringify(args).replace(/"/g, '')}`);
  console.log(`${colors.yellow}└──────────────────────────────────────────────────────────┘${colors.reset}`);

  const answer = await askQuestion(`Approve execution? [y/N]: `);
  if (answer === null) return false;
  return answer.trim().toLowerCase() === 'y';
}

/**
 * Safe wrapper around rl.question: never throws after stdin closes.
 */
function askQuestion(prompt) {
  return new Promise((resolve) => {
    ensureRL();
    if (rl.closed) return resolve(null);
    rl.question(prompt, resolve);
  });
}

/* ── Core agentic loop ────────────────────────────────────────────────── */

let toolCallsInTurn = 0;
const MAX_TOOL_CALLS_PER_TURN = 40;

async function runAgentStep() {
  trimHistory();
  console.log(`\n${colors.blue}[SYSTEM]: Thinking...${colors.reset}`);

  try {
    const response = await queryLLM(chatHistory);
    chatHistory.push({ role: 'assistant', content: response });
    appendHistory(chatHistory[chatHistory.length - 1]);

    const toolCall = parseToolCall(response);
    if (toolCall) {
      if (toolCallsInTurn >= MAX_TOOL_CALLS_PER_TURN) {
        console.log(`${colors.red}[SYSTEM]: Too many tool calls in one turn (${MAX_TOOL_CALLS_PER_TURN}); stopping.${colors.reset}`);
        promptUser();
        return;
      }
      toolCallsInTurn++;

      const cleanText = response.replace(/```json[\s\S]*?```/, '').trim();
      if (cleanText && !CONFIG.stream) {
        console.log(`\n${colors.magenta}AMcoli Agent:${colors.reset} ${cleanText}`);
      }

      const approved = await askApproval(toolCall.tool, toolCall.args || {});
      if (approved) {
        console.log(`${colors.green}[SYSTEM]: Running tool ${toolCall.tool}...${colors.reset}`);
        const result = await Tools[toolCall.tool](toolCall.args || {});
        console.log(`${colors.green}[SYSTEM]: Tool output returned (Length: ${result.length} characters)${colors.reset}`);
        chatHistory.push({ role: 'user', content: `Tool Execution Result:\n\`\`\`\n${result}\n\`\`\`` });
        appendHistory(chatHistory[chatHistory.length - 1]);
        await runAgentStep();
      } else {
        console.log(`${colors.red}[SYSTEM]: Execution denied by user.${colors.reset}`);
        chatHistory.push({
          role: 'user',
          content: `Tool Execution Result:\nError: Execution of tool '${toolCall.tool}' denied by the user.`
        });
        appendHistory(chatHistory[chatHistory.length - 1]);
        await runAgentStep();
      }
    } else {
      if (!CONFIG.stream) {
        console.log(`\n${colors.magenta}AMcoli Agent:${colors.reset} ${response}`);
      }
      promptUser();
    }
  } catch (err) {
    console.error(`\n${colors.red}[ERROR]: ${err.message}${colors.reset}`);
    promptUser();
  }
}

/* ── Slash commands ───────────────────────────────────────────────────── */

function handleSlashCommand(input) {
  const [cmd, ...rest] = input.slice(1).split(/\s+/);
  const value = rest.join(' ').trim();
  switch (cmd) {
    case 'help':
    case '?':
      console.log(`${colors.cyan}
  /help            Show this help
  /clear           Clear the chat history (current session)
  /reset           Clear the chat history AND the persisted session file
  /model <name>    Switch the inference model
  /api <url>       Switch the OpenAI-compatible API endpoint
  /quit, /exit     Exit AMcoli Agent
  (everything else is sent to the model as a prompt)${colors.reset}`);
      return true;
    case 'clear':
      chatHistory.splice(1);
      console.log(`${colors.green}[SYSTEM]: Chat history cleared.${colors.reset}`);
      return true;
    case 'reset':
      chatHistory.splice(1);
      clearHistoryFile();
      console.log(`${colors.green}[SYSTEM]: Session reset (memory + history file).${colors.reset}`);
      return true;
    case 'model':
      if (!value) {
        console.log(`${colors.yellow}Current model: ${CONFIG.model}${colors.reset}`);
      } else {
        CONFIG.model = value;
        console.log(`${colors.green}[SYSTEM]: Model switched to ${value}.${colors.reset}`);
      }
      return true;
    case 'api':
      if (!value) {
        console.log(`${colors.yellow}Current API endpoint: ${CONFIG.apiUrl}${colors.reset}`);
      } else {
        CONFIG.apiUrl = value.replace(/\/$/, '');
        console.log(`${colors.green}[SYSTEM]: API endpoint switched to ${CONFIG.apiUrl}.${colors.reset}`);
      }
      return true;
    case 'quit':
    case 'exit':
      console.log(`\n${colors.yellow}Exiting AMcoli Agentic CLI. Bye!${colors.reset}\n`);
      process.exit(0);
      return true;
    default:
      return false;
  }
}

/**
 * Prompt the user for input
 */
async function promptUser() {
  const input = await askQuestion(`\n${colors.green}You (type /help or 'q' to quit) > ${colors.reset}`);
  if (input === null) return; // stdin closed; 'close' handler will exit

  const text = input.trim();
  if (text.toLowerCase() === 'q') {
    console.log(`\n${colors.yellow}Exiting AMcoli Agentic CLI. Bye!${colors.reset}\n`);
    process.exit(0);
  }
  if (!text) return promptUser();
  if (text.startsWith('/')) {
    if (handleSlashCommand(text)) return promptUser(); // re-prompt after slash command
  }

  chatHistory.push({ role: 'user', content: text });
  appendHistory(chatHistory[chatHistory.length - 1]);
  toolCallsInTurn = 0;
  runAgentStep();
}

/* ── Main entry point ─────────────────────────────────────────────────── */

function startAgent(options = {}) {
  if (options.apiUrl) CONFIG.apiUrl = String(options.apiUrl).replace(/\/$/, '');
  if (options.model) CONFIG.model = options.model;
  if (options.apiKey) CONFIG.apiKey = options.apiKey;
  if (options.cwd) {
    CONFIG.cwd = path.resolve(options.cwd);
    if (!fs.existsSync(CONFIG.cwd)) {
      console.error(`${colors.red}Error: working directory does not exist: ${CONFIG.cwd}${colors.reset}`);
      process.exit(1);
    }
  }
  if (options.yes) CONFIG.autoApprove = true;
  if (options.auto) CONFIG.autoReadonly = true;
  if (options.stream) CONFIG.stream = true;

  ensureRL();
  loadHistory();

  const mode = CONFIG.autoApprove
    ? `${colors.yellow}AUTO-APPROVE ALL${colors.reset}`
    : CONFIG.autoReadonly
      ? `${colors.yellow}AUTO-APPROVE READ-ONLY${colors.reset}`
      : `${colors.bright}${colors.yellow}EXPLICIT CONFIRMATION FOR ALL TOOL RUNS${colors.reset}`;

  console.log(`\n${colors.gray}================================================================================${colors.reset}`);
  for (const line of BANNER_LOGO) console.log(line);
  console.log(`${colors.gray}================================================================================${colors.reset}`);
  console.log(` ${colors.bright}${colors.cyan}AMCOLI — AGENTIC WORKFLOW INTERFACE${colors.reset}`);
  console.log(`${colors.gray}================================================================================${colors.reset}`);
  console.log(`  LLM Server      : ${colors.green}${CONFIG.apiUrl}${colors.reset}`);
  console.log(`  Inference Model : ${colors.green}${CONFIG.model}${colors.reset}`);
  console.log(`  Working Dir     : ${colors.green}${CONFIG.cwd}${colors.reset}`);
  console.log(`  Tools Loaded    : ${colors.green}list_dir, read_file, search_files, get_system_info, write_file, edit_file, run_command, run_amcoli${colors.reset}`);
  console.log(`  Security Mode   : ${mode}${colors.reset}`);
  console.log(`  Streaming       : ${colors.green}${CONFIG.stream ? 'ON' : 'OFF'}${colors.reset}`);
  console.log(`${colors.gray}================================================================================\n${colors.reset}`);

  console.log(`${colors.cyan}Examples of what you can ask me to do:${colors.reset}`);
  console.log(` - "Find all C++ files here and explain the cache eviction policy in llama-moe-cache.cpp"`);
  console.log(` - "Search for every place GGUF magic is parsed and summarize the byte layout"`);
  console.log(` - "Run amcoli check and tell me which registry URLs are broken"`);
  console.log(` - "Compile the project using CMake and run the unit tests"`);
  console.log(` - "Check the git status and show me what was changed recently"`);

  promptUser();
}

module.exports = { startAgent };

// Test-only: expose internals when AMCOLI_AGENT_TEST=1 (used by CI/tests)
if (process.env.AMCOLI_AGENT_TEST === '1') {
  module.exports.tools = Tools;
  module.exports.config = CONFIG;
  module.exports.parseToolCall = parseToolCall;
  module.exports.buildSystemPrompt = buildSystemPrompt;
}
