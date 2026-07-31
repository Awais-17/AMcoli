#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const binaryName = os.platform() === 'win32' ? 'amcoli.exe' : 'amcoli';
const binaryPath = path.join(__dirname, binaryName);

const args = process.argv.slice(2);

// Check if user requested the agentic mode
if (args[0] === 'agent') {
  const agent = require('../lib/agent');
  if (args[1] === '--help' || args[1] === '-h' || args[1] === 'help') {
    console.log('Usage: amcoli agent [options]');
    console.log('');
    console.log('Interactive agentic CLI: an AI coding assistant that runs locally in your terminal.');
    console.log('');
    console.log('Options:');
    console.log('  -u, --api-url <url>   OpenAI-compatible API endpoint (default: http://localhost:11434/v1)');
    console.log('  -m, --model <name>    Model name (default: qwen2.5-coder:7b)');
    console.log('  -k, --api-key <key>   API key for the endpoint');
    console.log('  --cwd <dir>           Working directory for the agent (default: current dir)');
    console.log('  -y, --yes             Auto-approve file modifications and command execution');
    console.log('      --auto            Auto-approve read-only tools, prompt for the rest');
    console.log('      --stream          Stream model output as it is generated');
    console.log('');
    console.log('Slash commands inside the session:');
    console.log('  /help        Show this help');
    console.log('  /clear       Clear chat history');
    console.log('  /reset       Clear chat history AND the persisted session file');
    console.log('  /model       Show current model');
    console.log('  /model <n>   Switch inference model');
    console.log('  /api         Show current API endpoint');
    console.log('  /api <url>   Switch API endpoint');
    console.log('  /quit, q     Exit the agent');
    process.exit(0);
  }
  const options = {};
  for (let i = 1; i < args.length; i++) {
    const a = args[i];
    if ((a === '--api-url' || a === '-u') && args[i + 1]) {
      options.apiUrl = args[i + 1];
      i++;
    } else if ((a === '--model' || a === '-m') && args[i + 1]) {
      options.model = args[i + 1];
      i++;
    } else if ((a === '--api-key' || a === '-k') && args[i + 1]) {
      options.apiKey = args[i + 1];
      i++;
    } else if ((a === '--cwd') && args[i + 1]) {
      options.cwd = args[i + 1];
      i++;
    } else if (a === '--yes' || a === '-y') {
      options.yes = true;
    } else if (a === '--auto') {
      options.auto = true;
    } else if (a === '--stream') {
      options.stream = true;
    } else {
      console.error(`\x1b[33mUnknown agent option: ${a}\x1b[0m`);
      console.error('Usage: amcoli agent [--api-url <url>] [--model <name>] [--api-key <key>] [--cwd <dir>] [--yes] [--auto] [--stream]');
      process.exit(1);
    }
  }
  agent.startAgent(options);
} else {
  // Otherwise, spawn the native binary
  if (!fs.existsSync(binaryPath)) {
    console.error('\x1b[31mError: AMcoli native binary not found at:\x1b[0m');
    console.error(`  ${binaryPath}`);
    console.error('\x1b[33mPlease re-run "npm install" or compile the project from source.\x1b[0m');
    process.exit(1);
  }

  const child = spawn(binaryPath, args, {
    stdio: 'inherit',
    shell: false
  });

  child.on('error', (err) => {
    console.error('\x1b[31mFailed to start AMcoli process:\x1b[0m', err.message);
    process.exit(1);
  });

  child.on('close', (code) => {
    process.exit(code !== null ? code : 0);
  });
}
