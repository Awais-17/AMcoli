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
  const options = {};
  for (let i = 1; i < args.length; i++) {
    if ((args[i] === '--api-url' || args[i] === '-u') && args[i + 1]) {
      options.apiUrl = args[i + 1];
      i++;
    } else if ((args[i] === '--model' || args[i] === '-m') && args[i + 1]) {
      options.model = args[i + 1];
      i++;
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
