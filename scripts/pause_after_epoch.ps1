param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][int]$Epoch,
    [Parameter(Mandatory = $true)][int]$TrainingProcessId,
    [Parameter(Mandatory = $true)][int]$TrainingWrapperProcessId,
    [Parameter(Mandatory = $true)][int]$QueueProcessId
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot "outputs\runs\$RunId"
$epochLog = Join-Path $runDir "epochs.jsonl"
$statusPath = Join-Path $runDir "status.json"
$queueStatusPath = Join-Path $repoRoot "outputs\queue_status.json"
$pausePath = Join-Path $runDir "pause.json"

while ($true) {
    $epochRecord = $null
    if (Test-Path $epochLog) {
        $records = @(Get-Content $epochLog | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        })
        $epochRecord = $records | Where-Object { $_.epoch -eq $Epoch } | Select-Object -Last 1
    }
    $bestAdapter = Join-Path $runDir "best_adapter\adapter_model.safetensors"
    $lastAdapter = Join-Path $runDir "last_adapter\adapter_model.safetensors"
    if ($null -ne $epochRecord -and (Test-Path $bestAdapter) -and (Test-Path $lastAdapter)) {
        break
    }
    if (-not (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue)) {
        throw "Training process exited before epoch $Epoch was checkpointed."
    }
    Start-Sleep -Seconds 2
}

# Stop the queue first so it cannot interpret the intentional trainer exit as a
# failure and launch another run. The model processes are then stopped only
# after both epoch adapters and the complete epoch record exist.
Stop-Process -Id $QueueProcessId -Force -ErrorAction SilentlyContinue
Stop-Process -Id $TrainingProcessId -Force -ErrorAction SilentlyContinue
Stop-Process -Id $TrainingWrapperProcessId -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$pausedAt = (Get-Date).ToUniversalTime().ToString("o")
[ordered]@{
    state = "paused"
    paused_after_epoch = $Epoch
    best_epoch = $epochRecord.epoch
    best_validation_loss = $epochRecord.validation_loss
    optimizer_step = $epochRecord.optimizer_step
    total_optimizer_steps = 3750
    resume_adapter = "last_adapter"
    pause_reason = "User requested a durable pause after epoch validation and checkpointing"
    paused_at_utc = $pausedAt
} | ConvertTo-Json | Set-Content $statusPath -Encoding utf8

[ordered]@{
    state = "paused"
    run_id = $RunId
    paused_after_epoch = $Epoch
    adapter_checkpoint = "last_adapter"
    best_adapter_checkpoint = "best_adapter"
    epoch_log = "epochs.jsonl"
    step_log = "steps.jsonl"
    paused_at_utc = $pausedAt
} | ConvertTo-Json | Set-Content $pausePath -Encoding utf8

[ordered]@{
    state = "paused"
    active_model = "llama"
    active_run_id = $RunId
    message = "Paused after epoch $Epoch validation and adapter checkpointing at user request"
    updated_at = $pausedAt
} | ConvertTo-Json | Set-Content $queueStatusPath -Encoding utf8
