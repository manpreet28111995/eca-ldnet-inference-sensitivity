param(
    [int[]]$Seeds = @(17, 43, 73),
    [string]$PythonExe,
    [switch]$AllowCpu,
    [switch]$Fixed256
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RepoVenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RepoWslPython = Join-Path $ProjectRoot '.venv\bin\python'
$PackageArchive = Join-Path $ProjectRoot 'kaggle_revision\eca_ldnet_reviewer_test_package.zip'
$RuntimeRoot = Join-Path $ProjectRoot 'evaluation_all_models_revisions\reviewer_eval_runtime'
$PackageRoot = Join-Path $RuntimeRoot 'reviewer_test_package'
$ResultRoot = Join-Path $ProjectRoot 'evaluation_all_models_revisions\reviewer_evaluation'
$EvalScript = Join-Path $ProjectRoot 'scripts\evaluate_paired.py'
$ImageExtensions = @('.png', '.jpg', '.jpeg', '.bmp')

function Convert-ToWslPath([string]$WindowsPath) {
    if ($WindowsPath -match '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
        return '/mnt/' + $Matches.drive.ToLowerInvariant() + '/' + ($Matches.rest -replace '\\', '/')
    }
    return $WindowsPath
}

if ($PythonExe) {
    $Python = (Resolve-Path -LiteralPath $PythonExe).Path
    $UseWsl = $false
} elseif (Test-Path -LiteralPath $RepoVenvPython -PathType Leaf) {
    $Python = $RepoVenvPython
    $UseWsl = $false
} elseif ($env:VIRTUAL_ENV -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $Python = (Get-Command python).Source
    $UseWsl = $false
} elseif (Test-Path -LiteralPath $RepoWslPython) {
    $WslExeCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $WslExeCommand) { throw 'The repository venv is a WSL venv, but wsl.exe is unavailable. Activate a Windows GPU venv or pass -PythonExe.' }
    $WslExe = $WslExeCommand.Source
    # wslpath is unreliable when PowerShell passes a Windows path containing
    # backslashes. Convert the resolved drive path explicitly to /mnt/<drive>.
    if ($ProjectRoot -match '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
        $WslProject = Convert-ToWslPath $ProjectRoot
    } else {
        $WslProject = (& $WslExe -- wslpath -a -- $ProjectRoot 2>$null).Trim()
    }
    if (-not $WslProject) { throw "Could not map the project path into WSL: $ProjectRoot" }
    $Python = '.venv/bin/python'
    $UseWsl = $true
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) { throw 'Python was not found. Activate your Windows GPU venv first or pass -PythonExe.' }
    $Python = $PythonCommand.Source
    $UseWsl = $false
}
if (-not $UseWsl -and -not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable missing: $Python" }
if (-not (Test-Path -LiteralPath $PackageArchive -PathType Leaf)) { throw "Reviewer checkpoint package missing: $PackageArchive" }
if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot 'package_manifest.json') -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
    Expand-Archive -LiteralPath $PackageArchive -DestinationPath $RuntimeRoot -Force
}
$ManifestPath = Join-Path $PackageRoot 'package_manifest.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Package manifest missing after extraction: $ManifestPath" }
New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null

Write-Host 'Checking the venv TensorFlow/GPU before evaluation...'
if ($UseWsl) {
    & $WslExe --cd $WslProject -- $Python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU')); assert tf.config.list_physical_devices('GPU'), 'No TensorFlow GPU detected'"
} else {
    $env:PYTHONPATH = $ProjectRoot
    & $Python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU')); assert tf.config.list_physical_devices('GPU'), 'No TensorFlow GPU detected'"
}
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowCpu) {
        throw 'TensorFlow/GPU preflight failed; WSL has no usable CUDA runtime. Fix WSL GPU support or rerun with -AllowCpu for a slow CPU-only evaluation.'
    }
    Write-Warning 'Continuing on CPU because -AllowCpu was specified. This may be substantially slower.'
}

$DatasetSpecs = @(
    [pscustomobject]@{ Name = 'I-HAZE'; Hazy = (Join-Path $ProjectRoot 'datasets\I-HAZE\test\hazy'); GT = (Join-Path $ProjectRoot 'datasets\I-HAZE\test\clear'); OfficialCount = 35 },
    [pscustomobject]@{ Name = 'NH-HAZE'; Hazy = (Join-Path $ProjectRoot 'datasets\NH-HAZE\test\hazy'); GT = (Join-Path $ProjectRoot 'datasets\NH-HAZE\test\clear'); OfficialCount = 55 }
)
foreach ($dataset in $DatasetSpecs) {
    if (-not (Test-Path -LiteralPath $dataset.Hazy -PathType Container)) { throw "Hazy folder missing: $($dataset.Hazy)" }
    if (-not (Test-Path -LiteralPath $dataset.GT -PathType Container)) { throw "Clear/GT folder missing: $($dataset.GT)" }
    $hazyFiles = @(Get-ChildItem -LiteralPath $dataset.Hazy -File | Where-Object { $ImageExtensions -contains $_.Extension.ToLowerInvariant() })
    $gtFiles = @(Get-ChildItem -LiteralPath $dataset.GT -File | Where-Object { $ImageExtensions -contains $_.Extension.ToLowerInvariant() })
    Write-Host ("{0}: {1} hazy test images, {2} clear images. This local test split is smaller than the {3}-pair full release." -f $dataset.Name, $hazyFiles.Count, $gtFiles.Count, $dataset.OfficialCount)
    if ($hazyFiles.Count -eq 0 -or $gtFiles.Count -eq 0) { throw "No test images found for $($dataset.Name)." }
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$models = @($manifest.models | Where-Object { $Seeds -contains [int]$_.seed })
if ($models.Count -eq 0) { throw "No package models found for requested seeds: $($Seeds -join ', ')" }
$expectedSeeds = @($models | ForEach-Object { [int]$_.seed } | Sort-Object -Unique)
Write-Host ("Running {0} checkpoints for seed(s): {1}" -f $models.Count, ($expectedSeeds -join ', '))
if ($Fixed256) {
    Write-Host 'Inference mode: fixed 256x256 GPU evaluation.'
} else {
    Write-Host 'Each image is scored at native dimensions using 512x512 tiles, overlap 96, tile batch 1.'
}
Write-Host 'All matched test pairs are scored; up to 10 deterministic sample predictions are saved per model/dataset.'

$ProgressPath = Join-Path $ResultRoot 'real_i_nh_progress.json'
$Completed = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
if (Test-Path -LiteralPath $ProgressPath -PathType Leaf) {
    $prior = Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json
    foreach ($key in @($prior.complete)) { [void]$Completed.Add([string]$key) }
}

$allTasks = $models.Count * $DatasetSpecs.Count
$taskNumber = 0
foreach ($model in $models) {
    $checkpoint = Join-Path $PackageRoot ([string]$model.checkpoint)
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Checkpoint missing: $checkpoint" }
    $actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$model.checkpoint_sha256).ToLowerInvariant()) { throw "Checkpoint SHA-256 mismatch: $checkpoint" }
    foreach ($dataset in $DatasetSpecs) {
        $taskNumber++
        $key = "paired::$($model.model_id)::$($dataset.Name)"
        $dest = Join-Path $ResultRoot (Join-Path 'paired' (Join-Path ([string]$model.model_id) $dataset.Name))
        $runManifest = Join-Path $dest 'paired_run_manifest.json'
        if ($Completed.Contains($key) -and (Test-Path -LiteralPath $runManifest -PathType Leaf)) {
            Write-Host ("[{0}/{1}] Already complete; keeping {2}" -f $taskNumber, $allTasks, $key)
            continue
        }

        $sampleDir = Join-Path $dest "tested_sample_$($dataset.Name)_seed42_max10"
        $evalArgs = @(
            $EvalScript, '--model', $checkpoint, '--variant', [string]$model.variant,
            '--hazy-dir', $dataset.Hazy, '--gt-dir', $dataset.GT,
            '--dataset', $dataset.Name, '--output', $dest,
            '--preprocessing', 'radiometric',
            '--bootstrap-samples', '5000', '--seed', '42',
            '--tested-sample-dir', $sampleDir, '--save-prediction-samples', '10'
        )
        if ($Fixed256) { $evalArgs += @('--size', '256') } else { $evalArgs += '--native-resolution' }
        Write-Host ("[{0}/{1}] Evaluating {2} on {3}" -f $taskNumber, $allTasks, $model.model_id, $dataset.Name)
        if ($UseWsl) {
            $wslEvalArgs = @()
            foreach ($arg in $evalArgs) {
                if ($arg -is [string] -and ($arg -eq $EvalScript -or $arg.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase))) {
                    $wslEvalArgs += (Convert-ToWslPath $arg)
                } else {
                    $wslEvalArgs += [string]$arg
                }
            }
            & $WslExe --cd $WslProject -- $Python @wslEvalArgs
        } else {
            & $Python @evalArgs
        }
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $key (exit code $LASTEXITCODE). Rerun this script to resume completed tasks." }
        [void]$Completed.Add($key)
        $progressObject = [ordered]@{ complete = @($Completed | Sort-Object); updated_utc = (Get-Date).ToUniversalTime().ToString('o') }
        $progressJson = $progressObject | ConvertTo-Json -Depth 5
        [System.IO.File]::WriteAllText($ProgressPath, $progressJson, [System.Text.UTF8Encoding]::new($false))
    }
}

# Refresh dataset summary tables without touching the existing SOTS/O-HAZE rows.
$newRows = @()
foreach ($model in $models) {
    foreach ($dataset in $DatasetSpecs) {
        $summaryPath = Join-Path $ResultRoot (Join-Path 'paired' (Join-Path ([string]$model.model_id) (Join-Path $dataset.Name 'paired_summary.csv')))
        if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) { continue }
        foreach ($row in (Import-Csv -LiteralPath $summaryPath)) {
            $newRows += [pscustomobject][ordered]@{
                family = [string]$model.family; variant = [string]$model.variant
                model_id = [string]$model.model_id; seed = [int]$model.seed
                dataset = $dataset.Name; metric = [string]$row.metric; n = [int]$row.n
                mean = [double]$row.mean; std = [double]$row.std
                bootstrap_ci95_low = [double]$row.bootstrap_ci95_low
                bootstrap_ci95_high = [double]$row.bootstrap_ci95_high
            }
        }
    }
}
$summaryPath = Join-Path $ResultRoot 'per_model_dataset_summary.csv'
$oldRows = @()
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    $oldRows = @(Import-Csv -LiteralPath $summaryPath | Where-Object { $_.dataset -notin @('I-HAZE', 'NH-HAZE') })
}
$combinedRows = @($oldRows) + @($newRows)
if ($combinedRows.Count) { $combinedRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8 }

$aggregateRows = @()
foreach ($group in ($combinedRows | Group-Object { "$($_.family)|$($_.variant)|$($_.dataset)|$($_.metric)" })) {
    $first = $group.Group[0]
    $values = @($group.Group | ForEach-Object { [double]$_.mean })
    $mean = ($values | Measure-Object -Average).Average
    $sd = ''
    if ($values.Count -gt 1) {
        $sumSq = 0.0
        foreach ($value in $values) { $sumSq += [Math]::Pow(($value - $mean), 2) }
        $sd = [Math]::Sqrt($sumSq / ($values.Count - 1))
    }
    $aggregateRows += [pscustomobject][ordered]@{
        family = [string]$first.family; variant = [string]$first.variant
        dataset = [string]$first.dataset; metric = [string]$first.metric
        seed_count = $values.Count; mean_of_seed_means = $mean
        sample_sd_across_seed_means = $sd
    }
}
if ($aggregateRows.Count) {
    $aggregateRows | Export-Csv -LiteralPath (Join-Path $ResultRoot 'across_seed_summary.csv') -NoTypeInformation -Encoding utf8
}

foreach ($dataset in $DatasetSpecs) {
    $datasetEvidence = Join-Path $ResultRoot $dataset.Name
    New-Item -ItemType Directory -Force -Path $datasetEvidence | Out-Null
    $rows = @($newRows | Where-Object { $_.dataset -eq $dataset.Name })
    if ($rows.Count) { $rows | Export-Csv -LiteralPath (Join-Path $datasetEvidence 'model_summary.csv') -NoTypeInformation -Encoding utf8 }
    $readme = @(
        "$($dataset.Name) reviewer evaluation evidence",
        "Split: local test folders only ($($dataset.OfficialCount) pairs in the full release; the attached local test split may contain fewer).",
        'All matched test pairs are scored. At most ten fixed-seed sampled predictions are saved per model.',
        'Inference: radiometric preprocessing; native dimensions; 512x512 tiles, overlap 96, tile batch 1.',
        'Per-model predictions, per-image scores, summaries and manifests are under paired/<model_id>/' + $dataset.Name + '/.'
    ) -join "`r`n"
    [System.IO.File]::WriteAllText((Join-Path $datasetEvidence 'README.txt'), $readme, [System.Text.UTF8Encoding]::new($false))
}

$mainProgressPath = Join-Path $ResultRoot 'progress.json'
$mainProgress = if (Test-Path -LiteralPath $mainProgressPath -PathType Leaf) {
    Get-Content -LiteralPath $mainProgressPath -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{ complete = @(); updated_unix = 0 }
}
$mergedProgressKeys = @(@($mainProgress.complete) + @($Completed | Sort-Object))
$mainProgress.complete = @($mergedProgressKeys | Sort-Object -Unique)
$mainProgress.updated_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$mainProgress | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $mainProgressPath -Encoding utf8

# Preserve the previous overall manifest and add the newly evaluated data paths/tasks.
$overallPath = Join-Path $ResultRoot 'evaluation_manifest.json'
if (Test-Path -LiteralPath $overallPath -PathType Leaf) {
    $overall = Get-Content -LiteralPath $overallPath -Raw | ConvertFrom-Json
    $priorDatasets = @($overall.paired_datasets | Where-Object { $_.name -notin @('I-HAZE', 'NH-HAZE') })
    $addedDatasets = @($DatasetSpecs | ForEach-Object { [pscustomobject]@{ name = $_.Name; hazy_dir = $_.Hazy; gt_dir = $_.GT; native_resolution = $true; size = $null } })
    $overall.paired_datasets = @($priorDatasets) + @($addedDatasets)
    $mergedTaskKeys = @(@($overall.completed_tasks) + @($Completed | Sort-Object))
    $overall.completed_tasks = @($mergedTaskKeys | Sort-Object -Unique)
    $overall | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $overallPath -Encoding utf8
}

Write-Host "Completed. Results root: $ResultRoot"
Write-Host 'Open I-HAZE and NH-HAZE folders for dataset-level tables; per-model details and sample outputs are under paired/<model_id>/<dataset>.'
