# Supervisor: keeps the training loop alive across hard process deaths.
# loop.py is resumable (reloads checkpoints/best.pt and the existing games
# buffer), so restarting simply continues. Each launch + exit is logged to
# logs/supervisor.log with a timestamp and exit code.
#
# Usage:   powershell -ExecutionPolicy Bypass -File run_training.ps1
# Stop:    delete logs/STOP  (the supervisor checks for it between restarts),
#          or Stop-Process the python child and create that file.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py        = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$superLog  = Join-Path $PSScriptRoot "logs\supervisor.log"
$stopFile  = Join-Path $PSScriptRoot "logs\STOP"
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "logs") | Out-Null

$loopArgs = @(
    "-m", "src.loop",
    "--iterations", "500",
    "--games-per-iter", "20",
    "--train-steps", "400",
    "--eval-games", "16",
    "--gen-sims", "100",
    "--eval-sims", "100",
    "--channels", "128",
    "--blocks", "10",
    "--batch-size", "256",
    "--log-every", "100",
    "--device", "cuda",
    "--log-level", "INFO"
)

function Log-Super($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $superLog -Value $line
}

if (Test-Path $stopFile) { Remove-Item $stopFile -Force }

$attempt = 0
while ($true) {
    if (Test-Path $stopFile) {
        Log-Super "STOP file present; supervisor exiting."
        break
    }
    $attempt++
    Log-Super "starting training (attempt $attempt)…"

    # Append child stdout/stderr so a crash traceback (faulthandler) is kept.
    & $py $loopArgs 1>> (Join-Path $PSScriptRoot "logs\training-run.out") `
                    2>> (Join-Path $PSScriptRoot "logs\training-run.err")
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Log-Super "training exited cleanly (code 0); supervisor done."
        break
    }

    Log-Super "training died (exit code $code); restarting in 15s."
    Start-Sleep -Seconds 15
}
