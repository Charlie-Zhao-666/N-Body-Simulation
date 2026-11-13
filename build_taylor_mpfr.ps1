param([string]$Compiler = 'g++')
$ErrorActionPreference = 'Stop'

# Use a 64-bit MSYS2 UCRT64 toolchain with MPFR and GMP development libraries.
$compilerPath = (Get-Command $Compiler -CommandType Application).Source
$target = Join-Path $PSScriptRoot 'nbody_taylor_mpfr.dll'
$buildDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'Codex/build-taylor-mpfr'
[IO.Directory]::CreateDirectory($buildDir) | Out-Null
$stagedName = 'nbody_taylor_mpfr-' + [Guid]::NewGuid().ToString('N') + '.dll'
$staged = Join-Path $buildDir $stagedName
$stagedRelative = Join-Path '../Codex/build-taylor-mpfr' $stagedName
$originalSearchPath = $env:Path
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:Path = (Split-Path -Parent $compilerPath) + ';' + $env:Path
    & $compilerPath -std=c++17 -O3 -Wall -Wextra -shared -static -DGMP_STATICLIB -s 'nbody_taylor_mpfr.cpp' -o $stagedRelative -lmpfr -lgmp
    if ($LASTEXITCODE -ne 0) { throw "MPFR backend build failed ($LASTEXITCODE)." }
    if ([IO.File]::Exists($target)) {
        [IO.File]::Replace($staged, $target, [System.Management.Automation.Language.NullString]::Value)
    } else {
        [IO.File]::Move($staged, $target)
    }
    Write-Output $target
} finally {
    $env:Path = $originalSearchPath
    Pop-Location
    if (Test-Path -LiteralPath $staged) { Remove-Item -LiteralPath $staged }
}
