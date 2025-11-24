# Wait for a training process to exit and extract trainer_state.json from the latest checkpoint
param(
    [int]$Pid = 16796,
    [string]$OutputDir = 'd:\Sarcasm Detection\outputs\baseline_full'
)
Write-Output "Watcher started at $(Get-Date -Format o) — waiting for PID $Pid to exit..."
try {
    Wait-Process -Id $Pid -ErrorAction Stop
    Write-Output "Process $Pid exited at $(Get-Date -Format o)"
} catch {
    Write-Output "Process $Pid was not found or already exited. Continuing to discovery step."
}

# Find latest checkpoint directory named checkpoint-<number>
$ck = Get-ChildItem -Path $OutputDir -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like 'checkpoint-*' } |
      Sort-Object { [int]($_.Name -replace 'checkpoint-','') } -Descending |
      Select-Object -First 1

if ($ck) {
    Write-Output "Latest checkpoint found: $($ck.FullName)"
    $trainer_state = Join-Path $ck.FullName 'trainer_state.json'
    if (Test-Path $trainer_state) {
        Write-Output "Contents of $trainer_state:" 
        Get-Content $trainer_state
    } else {
        Write-Output "trainer_state.json not found inside checkpoint. Looking for trainer_state.json in output dir..."
        $ts2 = Join-Path $OutputDir 'trainer_state.json'
        if (Test-Path $ts2) { Get-Content $ts2 } else { Write-Output 'No trainer_state.json found.' }
    }
} else {
    Write-Output "No checkpoint-* directories found under $OutputDir. Listing files instead:" 
    Get-ChildItem $OutputDir -File | Select-Object Name, Length, LastWriteTime
}

Write-Output "Watcher finished at $(Get-Date -Format o)"
