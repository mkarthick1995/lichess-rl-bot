# Supervisor: keeps the training loop alive across hard process deaths.
# loop.py is resumable (checkpoints/state.json + latest.pt + the games
# buffer), so restarting simply continues. Each launch + exit is logged to
# logs/supervisor.log with a timestamp and exit code.
#
# Usage:   powershell -ExecutionPolicy Bypass -File run_training.ps1
# Stop:    create logs/STOP — the loop saves and exits cleanly within a few
#          seconds, and the supervisor then exits too. (Starting the
#          supervisor again deletes logs/STOP.)
#
# Only one supervisor can run at a time (named mutex), so it is safe to start
# from a logon task and by hand. With -AtLogon (used by the scheduled task)
# an existing logs/STOP is respected instead of cleared, so a deliberate stop
# survives reboots.

param([switch]$AtLogon)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py        = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$superLog  = Join-Path $PSScriptRoot "logs\supervisor.log"
$stopFile  = Join-Path $PSScriptRoot "logs\STOP"
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "logs") | Out-Null

$loopArgs = @(
    "-m", "src.loop",
    "--workers", "7",
    "--parallel", "144",
    "--gen-sims", "100",
    "--eval-sims", "100",
    "--batch-size", "512",
    "--sample-ratio", "4",
    "--eval-every", "3000", "--eval-games", "160",
    "--promote-threshold", "0.55",
    "--channels", "128",
    "--blocks", "10",
    "--device", "cuda",
    "--log-level", "INFO"
)

function Log-Super($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $superLog -Value $line
}

$mutex = New-Object System.Threading.Mutex($false, "Global\lichess-rl-bot-training")
try {
    if (-not $mutex.WaitOne(0)) {
        Log-Super "another supervisor is already running; exiting."
        exit 0
    }
} catch [System.Threading.AbandonedMutexException] {
    # previous supervisor died without releasing; we now own it
}

if (Test-Path $stopFile) {
    if ($AtLogon) {
        Log-Super "logon start skipped: logs/STOP present (run the script by hand to resume)."
        $mutex.ReleaseMutex()
        exit 0
    }
    Remove-Item $stopFile -Force
}

$attempt = 0
while ($true) {
    if (Test-Path $stopFile) {
        Log-Super "STOP file present; supervisor exiting."
        break
    }
    $attempt++
    Log-Super "starting training (attempt $attempt)..."

    # Start-Process (not `& $py ... 2>>`): in Windows PowerShell, piping a
    # native program's stderr turns each line into an error record, which with
    # ErrorActionPreference=Stop can kill this supervisor — and the child with
    # it. Per-attempt files keep any crash traceback (faulthandler) around.
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $proc = Start-Process -FilePath $py -ArgumentList $loopArgs -NoNewWindow -Wait -PassThru `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput (Join-Path $PSScriptRoot "logs\training-run-$stamp.out") `
        -RedirectStandardError  (Join-Path $PSScriptRoot "logs\training-run-$stamp.err")
    $code = $proc.ExitCode

    if ($code -eq 0) {
        Log-Super "training exited cleanly (code 0); supervisor done."
        break
    }

    Log-Super "training died (exit code $code); restarting in 15s."
    Start-Sleep -Seconds 15
}
$mutex.ReleaseMutex()
