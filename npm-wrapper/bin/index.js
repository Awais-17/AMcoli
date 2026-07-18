#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const binaryName = os.platform() === 'win32' ? 'amcoli.exe' : 'amcoli';
const binaryPath = path.join(__dirname, binaryName);

if (!fs.existsSync(binaryPath)) {
  console.error('\x1b[31mError: AMcoli native binary not found at:\x1b[0m');
  console.error(`  ${binaryPath}`);
  console.error('\x1b[33mPlease re-run "npm install" or compile the project from source.\x1b[0m');
  process.exit(1);
}

// Forward CLI arguments (excluding 'node' and the script path) and inherit I/O streams
const child = spawn(binaryPath, process.argv.slice(2), {
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
