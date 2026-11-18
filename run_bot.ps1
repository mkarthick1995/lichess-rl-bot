# Supervisor for the Lichess bot service: keeps mercilessBot online,
# accepting bullet challenges (it never challenges anyone itself), and
# restarts it after a crash or a network outage at startup. The bot plays
# with checkpoints/v2/bot.pt (the network that passed the latest clocked
# bullet gate) and reloads it between games.
#
# Usage:   powershell -ExecutionPolicy Bypass -File run_bot.ps1
# Stop:    create logs/STOP_BOT. The bot finishes its current game, then the
#          bot and this supervisor exit. With -AtLogon, an existing STOP_BOT
#          is respected (the bot stays off across reboots); a manual start
#          clears it.
#
# Launched at logon by the scheduled task "lichess-rl-bot bot" and the
# Startup-folder shortcut "lichess-rl-bot bot.lnk"; a named mutex lets only one
# run.

param([switch]$AtLogon)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
try { [Console]::TreatControlCAsInput = $true } catch { }

$py       = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$superLog = Join-Path $PSScriptRoot "logs\bot-supervisor.log"
$stopFile = Join-Path $PSScriptRoot "logs\STOP_BOT"
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "logs") | Out-Null
$botArgs = @("-m", "src.lichess.bot")

function Log-Super($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $superLog -Value $line
}

$mutex = New-Object System.Threading.Mutex($false, "Global\lichess-rl-bot-bot")
try {
    if (-not $mutex.WaitOne(0)) {
        Log-Super "another bot supervisor is already running; exiting."
        exit 0
    }
} catch [System.Threading.AbandonedMutexException] { }

if (Test-Path $stopFile) {
    if ($AtLogon) {
        Log-Super "logon start skipped: logs/STOP_BOT present (run the script by hand to resume)."
        $mutex.ReleaseMutex()
        exit 0
    }
    Remove-Item $stopFile -Force
}

$attempt = 0
$delay = 30
while ($true) {
    if (Test-Path $stopFile) {
        Log-Super "STOP_BOT present; supervisor exiting."
        break
    }
    $attempt++
    Log-Super "starting bot (attempt $attempt)..."
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $t0 = Get-Date
    $proc = Start-Process -FilePath $py -ArgumentList $botArgs -WindowStyle Hidden -Wait -PassThru `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput (Join-Path $PSScriptRoot "logs\bot-run-$stamp.out") `
        -RedirectStandardError  (Join-Path $PSScriptRoot "logs\bot-run-$stamp.err")
    $code = $proc.ExitCode
    if ($code -eq 0 -and (Test-Path $stopFile)) {
        Log-Super "bot stopped cleanly; supervisor done."
        break
    }
    # Back off when it dies quickly (e.g. no network after boot); reset after a long run.
    if (((Get-Date) - $t0).TotalMinutes -gt 30) { $delay = 30 } else { $delay = [Math]::Min(600, $delay * 2) }
    Log-Super "bot exited (code $code); restarting in $delay s."
    Start-Sleep -Seconds $delay
}
$mutex.ReleaseMutex()
