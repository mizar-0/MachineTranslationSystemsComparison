param(
    [string]$Mt5RunId = "mt5_small_20261029_r3",
    [int]$WaitTimeoutHours = 4
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$queueStatusPath = Join-Path $repoRoot "outputs\queue_status.json"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$hf = Join-Path $repoRoot ".venv\Scripts\hf.exe"
$logs = Join-Path $repoRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Write-QueueStatus {
    param([string]$State, [string]$ActiveModel, [string]$Message, [string]$RunId = "")
    [ordered]@{
        state = $State
        active_model = $ActiveModel
        active_run_id = $RunId
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -Path $queueStatusPath -Encoding utf8
}

function Read-RunStatus {
    param([string]$RunId)
    $path = Join-Path $repoRoot "outputs\runs\$RunId\status.json"
    if (-not (Test-Path $path)) { return $null }
    try { return Get-Content $path -Raw | ConvertFrom-Json } catch { return $null }
}

function Assert-CompletedRun {
    param([string]$RunId)
    $status = Read-RunStatus $RunId
    if ($null -eq $status -or $status.state -ne "completed") {
        throw "Run $RunId did not reach completed state."
    }
    $predictions = Join-Path $repoRoot "outputs\runs\$RunId\test_predictions.jsonl"
    $summary = Join-Path $repoRoot "outputs\runs\$RunId\summary.json"
    if (-not (Test-Path $summary) -or -not (Test-Path $predictions)) {
        throw "Run $RunId is missing its summary or test predictions."
    }
    $predictionCount = (Get-Content $predictions).Count
    if ($predictionCount -ne 2507) {
        throw "Run $RunId has $predictionCount predictions; expected 2507."
    }
}

function Resolve-RunId {
    param([string]$BaseRunId)
    if (-not (Test-Path (Join-Path $repoRoot "outputs\runs\$BaseRunId"))) {
        return $BaseRunId
    }
    $existing = Read-RunStatus $BaseRunId
    if ($null -ne $existing -and $existing.state -eq "completed") {
        return $BaseRunId
    }
    $revision = 2
    while (Test-Path (Join-Path $repoRoot "outputs\runs\${BaseRunId}_r$revision")) {
        $revision += 1
    }
    return "${BaseRunId}_r$revision"
}

function Ensure-ModelSnapshot {
    param([string]$ModelId, [string]$LocalPath, [string]$LogName)
    $modelFile = Join-Path $LocalPath "model.safetensors"
    if (Test-Path $modelFile) { return }
    Write-QueueStatus "downloading" $ModelId "Downloading model snapshot to $LocalPath"
    & $hf download $ModelId --local-dir $LocalPath *> (Join-Path $logs $LogName)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $modelFile)) {
        throw "Download failed or model.safetensors is missing for $ModelId."
    }
}

function Start-CausalRun {
    param(
        [string]$Family,
        [string]$ModelId,
        [string]$LocalPath,
        [string]$BaseRunId,
        [int]$Epochs
    )
    $runId = Resolve-RunId $BaseRunId
    $existing = Read-RunStatus $runId
    if ($null -ne $existing -and $existing.state -eq "completed") {
        Assert-CompletedRun $runId
        return $runId
    }
    $stdout = Join-Path $logs "$runId.stdout.log"
    $stderr = Join-Path $logs "$runId.stderr.log"
    Write-QueueStatus "training" $Family "Starting report-grade LoRA training" $runId
    # Native programs legitimately use stderr for warnings. Temporarily keep
    # those warnings from becoming terminating PowerShell ErrorRecords and rely
    # on the process exit code for success/failure instead.
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python "src\train_causal_lora.py" `
        --model $Family `
        --model-id $ModelId `
        --model-path $LocalPath `
        --run-id $runId `
        --epochs $Epochs `
        --patience 1 `
        --batch-size 1 `
        --gradient-accumulation 16 `
        --validation-batch-size 2 `
        --generation-batch-size 2 `
        --max-source-length 96 `
        --max-target-length 96 `
        --log-every 100 `
        1> $stdout 2> $stderr
    $trainingExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    if ($trainingExitCode -ne 0) {
        throw "$Family training exited with code $trainingExitCode."
    }
    Assert-CompletedRun $runId
    return $runId
}

try {
    Write-QueueStatus "waiting" "mt5" "Waiting for mT5 training and evaluation" $Mt5RunId
    $deadline = (Get-Date).AddHours($WaitTimeoutHours)
    while ((Get-Date) -lt $deadline) {
        $mt5Status = Read-RunStatus $Mt5RunId
        if ($null -ne $mt5Status -and $mt5Status.state -eq "completed") { break }
        Start-Sleep -Seconds 30
    }
    Assert-CompletedRun $Mt5RunId

    $llamaId = "unsloth/Llama-3.2-1B"
    $llamaPath = Join-Path $repoRoot "models\llama-3.2-1b"
    Ensure-ModelSnapshot $llamaId $llamaPath "llama_download.queue.log"
    $llamaRun = Start-CausalRun "llama" $llamaId $llamaPath "llama_lora_20261030" 2

    $gemmaId = "unsloth/gemma-3-270m-it"
    $gemmaPath = Join-Path $repoRoot "models\gemma-3-270m-it"
    Ensure-ModelSnapshot $gemmaId $gemmaPath "gemma_download.queue.log"
    $gemmaRun = Start-CausalRun "gemma" $gemmaId $gemmaPath "gemma_lora_20261030" 2

    Write-QueueStatus "completed" "none" "All remaining model runs completed and validated: $llamaRun, $gemmaRun"
} catch {
    Write-QueueStatus "failed" "unknown" $_.Exception.Message
    $_ | Out-File (Join-Path $logs "training_queue.error.log") -Append
    exit 1
}
