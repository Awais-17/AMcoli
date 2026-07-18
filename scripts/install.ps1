# install.ps1 — AMcoli Installer for Windows
# Run via: irm https://raw.githubusercontent.com/Awais-17/AMcoli/main/scripts/install.ps1 | iex

$installDir = "$HOME\.amcoli"
$binDir = "$installDir\bin"
$exePath = "$binDir\amcoli.exe"
$repo = "Awais-17/AMcoli"

Write-Host "=========================================" -ForegroundColor Green
Write-Host "         AMcoli Windows Installer         " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# 1. Create installation directories
if (-not (Test-Path -Path $binDir)) {
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
}

$downloaded = $false

# 2. Try downloading pre-compiled binary from GitHub Releases
Write-Host "Checking for precompiled releases..." -ForegroundColor Cyan
$latestReleaseUrl = "https://api.github.com/repos/$repo/releases/latest"

try {
    # Set SecurityProtocol to TLS 1.2/1.3
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
    
    $release = Invoke-RestMethod -Uri $latestReleaseUrl -UseBasicParsing -ErrorAction Stop
    $asset = $release.assets | Where-Object { $_.name -like "*windows*.zip" -or $_.name -like "*win64*.zip" -or $_.name -like "*win-x64*.zip" } | Select-Object -First 1
    
    if ($asset) {
        $zipPath = "$installDir\amcoli-latest.zip"
        Write-Host "Downloading precompiled binary: $($asset.name)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -UseBasicParsing
        
        Write-Host "Extracting archive to $binDir..." -ForegroundColor Cyan
        Expand-Archive -Path $zipPath -DestinationPath $binDir -Force
        Remove-Item -Path $zipPath -Force
        $downloaded = $true
    } else {
        Write-Host "No Windows release asset found in the latest release." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not fetch precompiled release (either repository is private or no release exists yet)." -ForegroundColor Yellow
}

# 3. Fallback: Build from source if precompiled download wasn't possible
if (-not $downloaded) {
    Write-Host "`nFalling back to compiling from source..." -ForegroundColor Yellow
    
    # Check for CMake
    $cmakeCheck = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCheck) {
        Write-Error "CMake is required to compile from source but was not found on PATH. Please install CMake and try again."
        exit 1
    }
    
    $localBuild = $false
    # Check if we are running the installer from within the active repository source tree
    if (Test-Path -Path ".\CMakeLists.txt") {
        Write-Host "Local source tree detected, compiling from current directory..." -ForegroundColor Cyan
        $srcDir = (Get-Item ".\").FullName
        $localBuild = $true
    } else {
        $srcDir = "$installDir\src"
        if (Test-Path -Path $srcDir) {
            Remove-Item -Path $srcDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path $srcDir | Out-Null
        
        Write-Host "Cloning AMcoli repository..." -ForegroundColor Cyan
        # Check if git is available
        $gitCheck = Get-Command git -ErrorAction SilentlyContinue
        if ($gitCheck) {
            git clone --depth 1 "https://github.com/$repo.git" $srcDir
        } else {
            # Fallback to downloading zip of main branch
            Write-Host "Git not found, downloading source zip..." -ForegroundColor Yellow
            $zipUrl = "https://github.com/$repo/archive/refs/heads/main.zip"
            $srcZip = "$installDir\src.zip"
            Invoke-WebRequest -Uri $zipUrl -OutFile $srcZip -UseBasicParsing
            Expand-Archive -Path $srcZip -DestinationPath $srcDir -Force
            Remove-Item -Path $srcZip -Force
            
            # Expand-Archive nests it in AMcoli-main directory
            $nestedDir = Get-ChildItem -Path $srcDir -Directory | Select-Object -First 1
            if ($nestedDir) {
                Move-Item -Path "$($nestedDir.FullName)\*" -Destination $srcDir -Force
                Remove-Item -Path $nestedDir.FullName -Force
            }
        }
    }
    
    # Compile
    Write-Host "Configuring build directory..." -ForegroundColor Cyan
    Push-Location $srcDir
    try {
        & cmake -B build -DCMAKE_BUILD_TYPE=Release
        Write-Host "Building AMcoli binary..." -ForegroundColor Cyan
        & cmake --build build --config Release
        
        $builtExe = "$srcDir\build\Release\amcoli.exe"
        if (Test-Path -Path $builtExe) {
            Copy-Item -Path $builtExe -Destination $binDir -Force
            # Copy DLLs
            $releaseDir = (Split-Path $builtExe)
            $dlls = @("llama.dll", "ggml.dll")
            foreach ($dll in $dlls) {
                $dllPath = Join-Path $releaseDir $dll
                $binReleaseDir = Join-Path $srcDir "build\bin\Release"
                if (-not (Test-Path -Path $dllPath)) {
                    $dllPath = Join-Path $binReleaseDir $dll
                }
                if (Test-Path -Path $dllPath) {
                    Copy-Item -Path $dllPath -Destination $binDir -Force
                    Unblock-File -Path (Join-Path $binDir $dll) -ErrorAction SilentlyContinue
                }
            }
            $downloaded = $true
            Write-Host "Successfully compiled AMcoli." -ForegroundColor Green
        } else {
            Write-Error "Build finished but 'amcoli.exe' was not found."
        }
    } catch {
        Write-Error "Compilation failed: $_"
    } finally {
        Pop-Location
        # Cleanup source code if we cloned it
        if (-not $localBuild) {
            Remove-Item -Path $srcDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# 4. Check for success and configure PATH
if ($downloaded -and (Test-Path -Path $exePath)) {
    # Unblock file (Method B from setup guide) to satisfy Windows Smart App Control / WDAC
    Unblock-File -Path $exePath -ErrorAction SilentlyContinue
    Write-Host "Unblocked amcoli.exe for local system execution policy." -ForegroundColor Green
    
    # Configure PATH persistently for the User
    $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -split ';' -notcontains $binDir) {
        $newPath = "$userPath;$binDir"
        # Clean double semicolons if any
        $newPath = $newPath -replace ';;', ';'
        [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Host "`n[PATH UPDATED] Added '$binDir' to your User PATH environment variable." -ForegroundColor Green
        Write-Host "Please close and restart your terminal for the changes to take effect." -ForegroundColor Yellow
    } else {
        Write-Host "`nAMcoli is already in your PATH." -ForegroundColor Green
    }
    
    Write-Host "`nAMcoli installed successfully!" -ForegroundColor Green
    Write-Host "Type 'amcoli run' in a new terminal to start." -ForegroundColor Green
} else {
    Write-Error "Installation failed. Please review output errors above."
}
