# Axil one-line installer (Windows) — needs only built-in PowerShell 5+.
#
#   irm https://raw.githubusercontent.com/FC4b/axildb/main/scripts/install.ps1 | iex
#
# Downloads the archive for this platform from the latest GitHub release and
# installs axil.exe into $AXIL_HOME\bin (default ~\.axil\bin), then adds that
# directory to the user PATH. The release binary statically links onnxruntime,
# so no separate DLL is needed.
$ErrorActionPreference = "Stop"

$repo = "FC4b/axildb"

$arch = "x86_64"
if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { $arch = "aarch64" }
$triple = "$arch-pc-windows-msvc"

$url = "https://github.com/$repo/releases/latest/download/axildb-$triple.zip"
Write-Host "axil-install: downloading $url"

$tmp = Join-Path $env:Temp ("axil-install-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp "axildb.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    $axilHome = if ($env:AXIL_HOME) { $env:AXIL_HOME } else { Join-Path $HOME ".axil" }
    $installDir = Join-Path $axilHome "bin"
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    Copy-Item (Join-Path $tmp "axildb-$triple\*") $installDir -Force
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

# Add to the user PATH (persisted), and this session's PATH.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $installDir) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$installDir", "User")
    Write-Host "axil-install: added $installDir to the user PATH (new terminals)"
}
$env:Path += ";$installDir"

Write-Host "axil-install: installed -> $installDir\axil.exe"
& (Join-Path $installDir "axil.exe") --version
