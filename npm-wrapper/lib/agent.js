const fs = require('fs');
const path = require('path');
const http = require('http');
const { exec } = require('child_process');
const readline = require('readline');

// Colored Output Helpers
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  red: '\x1b[31m',
  bgGray: '\x1b[100m'
};

// Default Configuration
const CONFIG = {
  apiUrl: process.env.AMCOLI_API_URL || 'http://127.0.0.1:11434/v1', // Defaults to local Ollama
  model: process.env.AMCOLI_MODEL || 'qwen-coder-32b',               // Defaults to Qwen Coder
};

// System prompt defining rules and tool specifications
const SYSTEM_PROMPT = `You are "AMcoli Agent", a powerful, autonomous coding assistant.
You help the user develop, debug, and understand their C++ and system code.
You run locally in a terminal on the user's computer.

To interact with the local system, you have access to a set of tools. You can invoke them by responding with a JSON code block.
You MUST write the JSON block exactly like this (do NOT wrap it in any other JSON keys, do NOT add text inside the same block):
\`\`\`json
{
  "tool": "tool_name",
  "args": {
    "arg1": "value"
  }
}
\`\`\`

You can call only ONE tool per turn. After you call a tool, the system will execute it and return the result to you in the next message. Do not make up results; wait for the system to execute the tool.

### Supported Tools:

1. \`list_dir\`: List the contents of a directory.
   - Arguments: \`{"path": "string"}\` (use "." for current directory)

2. \`read_file\`: Read the contents of a specific file.
   - Arguments: \`{"path": "string"}\`

3. \`write_file\`: Create or overwrite a file with new content.
   - Arguments: \`{"path": "string", "content": "string"}\`

4. \`run_command\`: Run a shell command (compiling, tests, git, etc.).
   - Arguments: \`{"command": "string"}\`

### Core Instructions:
- Analyze the user's request. If you need to view files, list directories, or build/run tests to answer, immediately call the appropriate tool.
- Explain your reasoning before calling a tool.
- Keep your changes precise.
- When finished, summarize your work and output a regular text response.`;

// Conversation State
const chatHistory = [
  { role: 'system', content: SYSTEM_PROMPT }
];

// Readline interface for prompt
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

/**
 * Perform API request to OpenAI-compatible endpoint
 */
function queryLLM(messages) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(CONFIG.apiUrl);
    const postData = JSON.stringify({
      model: CONFIG.model,
      messages: messages,
      temperature: 0.2
    });

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
      path: `${parsedUrl.pathname}/chat/completions`.replace(/\/\//g, '/'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            const json = JSON.parse(data);
            resolve(json.choices[0].message.content);
          } catch (e) {
            reject(new Error(`Failed to parse API response: ${e.message}. Response was: ${data}`));
          }
        } else {
          reject(new Error(`API returned status ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', (e) => reject(e));
    req.write(postData);
    req.end();
  });
}

/**
 * Tool Implementations
 */
const Tools = {
  list_dir: ({ path: dirPath }) => {
    return new Promise((resolve) => {
      const resolved = path.resolve(dirPath || '.');
      fs.readdir(resolved, { withFileTypes: true }, (err, files) => {
        if (err) return resolve(`Error listing directory: ${err.message}`);
        const list = files.map(f => `${f.isDirectory() ? '[DIR] ' : '[FILE]'} ${f.name}`).join('\n');
        resolve(list || '(Directory is empty)');
      });
    });
  },

  read_file: ({ path: filePath }) => {
    return new Promise((resolve) => {
      const resolved = path.resolve(filePath);
      fs.readFile(resolved, 'utf8', (err, data) => {
        if (err) return resolve(`Error reading file: ${err.message}`);
        resolve(data);
      });
    });
  },

  write_file: ({ path: filePath, content }) => {
    return new Promise((resolve) => {
      const resolved = path.resolve(filePath);
      const parentDir = path.dirname(resolved);
      fs.mkdirSync(parentDir, { recursive: true });
      fs.writeFile(resolved, content, 'utf8', (err) => {
        if (err) return resolve(`Error writing file: ${err.message}`);
        resolve(`Successfully wrote file to ${filePath}`);
      });
    });
  },

  run_command: ({ command }) => {
    return new Promise((resolve) => {
      exec(command, (err, stdout, stderr) => {
        let output = '';
        if (stdout) output += stdout;
        if (stderr) output += `\n[STDERR]\n${stderr}`;
        if (err) output += `\n[ERROR: Code ${err.code}]\n${err.message}`;
        resolve(output.trim() || '(Command ran successfully with no output)');
      });
    });
  }
};

/**
 * Prompt user for yes/no approval of tool execution
 */
function askApproval(toolName, args) {
  return new Promise((resolve) => {
    console.log(`\n${colors.yellow}┌─── TOOL CALL APPROVAL REQUIRED ──────────────────────────┐${colors.reset}`);
    console.log(`${colors.yellow}│${colors.reset} Tool   : ${colors.bright}${colors.cyan}${toolName}${colors.reset}`);
    console.log(`${colors.yellow}│${colors.reset} Args   : ${JSON.stringify(args, null, 2).replace(/\n/g, `\n${colors.yellow}│${colors.reset} `)}`);
    console.log(`${colors.yellow}└──────────────────────────────────────────────────────────┘${colors.reset}`);
    
    rl.question(`Approve execution? [y/N]: `, (answer) => {
      const approved = answer.trim().toLowerCase() === 'y';
      resolve(approved);
    });
  });
}

/**
 * Parse JSON block out of assistant markdown response
 */
function parseToolCall(content) {
  const jsonRegex = /```json\s*([\s\S]*?)\s*```/;
  const match = content.match(jsonRegex);
  if (match) {
    try {
      const json = JSON.parse(match[1].trim());
      if (json.tool && Tools[json.tool]) {
        return json;
      }
    } catch (e) {
      // Failed to parse, treat as regular text
    }
  }
  return null;
}

/**
 * Core Agentic Loop Step
 */
async function runAgentStep() {
  console.log(`\n${colors.blue}[SYSTEM]: Thinking...${colors.reset}`);
  
  try {
    const response = await queryLLM(chatHistory);
    chatHistory.push({ role: 'assistant', content: response });

    // Print assistant response
    const toolCall = parseToolCall(response);
    if (toolCall) {
      // Print assistant reasoning/thought output, stripping out the tool JSON block
      const cleanText = response.replace(/```json[\s\S]*?```/, '').trim();
      if (cleanText) {
        console.log(`\n${colors.magenta}AMcoli Agent:${colors.reset} ${cleanText}`);
      }

      // Handle Tool Execution Flow
      const approved = await askApproval(toolCall.tool, toolCall.args || {});
      if (approved) {
        console.log(`${colors.green}[SYSTEM]: Running tool ${toolCall.tool}...${colors.reset}`);
        const result = await Tools[toolCall.tool](toolCall.args || {});
        
        console.log(`${colors.green}[SYSTEM]: Tool output returned (Length: ${result.length} characters)${colors.reset}`);
        chatHistory.push({
          role: 'user',
          content: `Tool Execution Result:\n\`\`\`\n${result}\n\`\`\``
        });
        
        // Loop immediately back to query the LLM with the tool output
        await runAgentStep();
      } else {
        console.log(`${colors.red}[SYSTEM]: Execution denied by user.${colors.reset}`);
        chatHistory.push({
          role: 'user',
          content: `Tool Execution Result:\nError: Execution of tool '${toolCall.tool}' denied by the user.`
        });
        
        // Loop back to LLM to explain the denial
        await runAgentStep();
      }
    } else {
      // Normal response, print it and wait for user's next prompt
      console.log(`\n${colors.magenta}AMcoli Agent:${colors.reset} ${response}`);
      promptUser();
    }
  } catch (err) {
    console.error(`\n${colors.red}[ERROR]: ${err.message}${colors.reset}`);
    promptUser();
  }
}

/**
 * Prompt User for input
 */
function promptUser() {
  rl.question(`\n${colors.green}You (Type 'q' to quit) > ${colors.reset}`, (input) => {
    const text = input.trim();
    if (text.toLowerCase() === 'q') {
      console.log(`\n${colors.yellow}Exiting AMcoli Agentic CLI. Bye!${colors.reset}\n`);
      rl.close();
      process.exit(0);
    }
    
    if (text) {
      chatHistory.push({ role: 'user', content: text });
      runAgentStep();
    } else {
      promptUser();
    }
  });
}

/**
 * Main Entry Point
 */
function startAgent(options = {}) {
  if (options.apiUrl) CONFIG.apiUrl = options.apiUrl;
  if (options.model) CONFIG.model = options.model;

  console.log(`\n================================================================================`);
  console.log(`               ${colors.bright}${colors.cyan}AMCOLI — AGENTIC WORKFLOW INTERFACE${colors.reset}`);
  console.log(`================================================================================`);
  console.log(`  Connected to LLM Server : ${colors.green}${CONFIG.apiUrl}${colors.reset}`);
  console.log(`  Active Inference Model  : ${colors.green}${CONFIG.model}${colors.reset}`);
  console.log(`  Tools Loaded            : ${colors.green}list_dir, read_file, write_file, run_command${colors.reset}`);
  console.log(`  Security Mode           : ${colors.bright}${colors.yellow}EXPLICIT CONFIRMATION REQUIRED FOR ALL TOOL RUNS${colors.reset}`);
  console.log(`================================================================================\n`);
  
  console.log(`${colors.cyan}Examples of what you can ask me to do:${colors.reset}`);
  console.log(` - "Find all C++ files here and explain the cache eviction policy in llama-moe-cache.cpp"`);
  console.log(` - "Compile the project using CMake and run the unit tests"`);
  console.log(` - "Check the git status and show me what was changed recently"`);
  
  promptUser();
}

module.exports = { startAgent };
