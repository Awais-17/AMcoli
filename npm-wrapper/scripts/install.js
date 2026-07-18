const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

const repo = 'Awais-17/AMcoli';
const platform = process.platform; // win32, linux, darwin
const arch = process.arch;         // x64, arm64

const binDir = path.join(__dirname, '..', 'bin');
if (!fs.existsSync(binDir)) {
  fs.mkdirSync(binDir, { recursive: true });
}

const binaryName = platform === 'win32' ? 'amcoli.exe' : 'amcoli';
const destPath = path.join(binDir, binaryName);

console.log('AMcoli Postinstall: Setting up native binary...');

// Helper to download a file, following redirects
function download(url, dest) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'AMcoli-NPM-Installer' } }, (res) => {
      if (res.statusCode === 302 || res.statusCode === 301) {
        download(res.headers.location, dest).then(resolve).catch(reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP Status ${res.statusCode}`));
        return;
      }
      const file = fs.createWriteStream(dest);
      res.pipe(file);
      file.on('finish', () => {
        file.close(() => resolve());
      });
    }).on('error', (err) => {
      reject(err);
    });
  });
}

async function tryDownloadPrecompiled() {
  // Map platform/arch to expected release asset names
  let osName = '';
  if (platform === 'win32') osName = 'windows';
  else if (platform === 'linux') osName = 'linux';
  else if (platform === 'darwin') osName = 'macos';

  let archName = arch === 'arm64' ? 'arm64' : 'x64';
  
  // Format: amcoli-windows-x64.exe or amcoli-linux-x64, etc.
  const assetName = `${osName}-${archName}${platform === 'win32' ? '.exe' : ''}`;
  const releaseUrl = `https://github.com/${repo}/releases/latest/download/${assetName}`;

  console.log(`Attempting to download precompiled binary from: ${releaseUrl}`);
  
  try {
    await download(releaseUrl, destPath);
    if (platform !== 'win32') {
      fs.chmodSync(destPath, 0o755);
    }
    console.log('\x1b[32mSuccessfully downloaded precompiled binary!\x1b[0m');
    return true;
  } catch (err) {
    console.warn(`Precompiled download unavailable (${err.message}).`);
    return false;
  }
}

function tryLocalBuild() {
  // Check if we are in the source repository (CMakeLists.txt is two levels up)
  const rootDir = path.join(__dirname, '..', '..');
  const cmakePath = path.join(rootDir, 'CMakeLists.txt');
  
  if (!fs.existsSync(cmakePath)) {
    console.log('Not in source repository. Fallback build skipped.');
    return false;
  }
  
  console.log('Source files found. Attempting local compilation via CMake...');
  
  try {
    // 1. Run cmake configure
    console.log('Configuring build...');
    execSync('cmake -B build -DCMAKE_BUILD_TYPE=Release', { cwd: rootDir, stdio: 'inherit' });
    
    // 2. Build executable
    console.log('Compiling binary...');
    execSync('cmake --build build --config Release', { cwd: rootDir, stdio: 'inherit' });
    
    // 3. Find compiled binary
    let builtBin = '';
    if (platform === 'win32') {
      builtBin = path.join(rootDir, 'build', 'Release', 'amcoli.exe');
    } else {
      builtBin = path.join(rootDir, 'build', 'amcoli');
    }
    
    if (fs.existsSync(builtBin)) {
      fs.copyFileSync(builtBin, destPath);
      if (platform !== 'win32') {
        fs.chmodSync(destPath, 0o755);
      }
      console.log('\x1b[32mSuccessfully compiled and installed local binary!\x1b[0m');
      return true;
    }
  } catch (e) {
    console.error('Local compilation failed:', e.message);
  }
  return false;
}

async function main() {
  // 1. Try to download precompiled binary
  let success = await tryDownloadPrecompiled();
  
  // 2. If download failed, attempt to compile from local source
  if (!success) {
    success = tryLocalBuild();
  }
  
  if (success) {
    console.log('\x1b[32mAMcoli installation setup completed successfully!\x1b[0m');
  } else {
    console.error('\x1b[31mCould not set up AMcoli binary.\x1b[0m');
    console.error('Please compile the C++ project manually and place the binary under "npm-wrapper/bin/".');
  }
}

main();
