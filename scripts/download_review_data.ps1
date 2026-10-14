$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DataRoot = Join-Path $ProjectRoot 'datasets\review_test'
$Downloads = Join-Path $DataRoot 'downloads'
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-FreeBytes([string]$Path) {
    $drive = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)).Substring(0,1)
    return [int64](Get-PSDrive -Name $drive -ErrorAction Stop).Free
}

function Expand-ZipResumable([string]$ZipPath, [string]$Destination) {
    $zip = $null
    try { $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath) }
    catch { throw "ZIP is incomplete or invalid; preserved without extraction: $ZipPath. $($_.Exception.Message)" }
    try {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        $root = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
        $todo = [Collections.Generic.List[object]]::new()
        [int64]$missingBytes = 0
        [int64]$replaceBytes = 0
        foreach ($entry in $zip.Entries) {
            if ([string]::IsNullOrEmpty($entry.Name)) { continue }
            $relative = $entry.FullName.Replace('/', [IO.Path]::DirectorySeparatorChar)
            $target = [IO.Path]::GetFullPath((Join-Path $root $relative))
            if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe archive path: $($entry.FullName)" }
            $valid = (Test-Path -LiteralPath $target -PathType Leaf) -and ((Get-Item -LiteralPath $target).Length -eq [int64]$entry.Length)
            if (-not $valid) {
                if (Test-Path -LiteralPath $target -PathType Leaf) { $replaceBytes += (Get-Item -LiteralPath $target).Length }
                $todo.Add([PSCustomObject]@{ Entry=$entry; Target=$target })
                $missingBytes += [int64]$entry.Length
            }
        }
        if ($todo.Count -eq 0) { Write-Host "Already complete; left untouched: $Destination"; return }
        $required = $missingBytes + $replaceBytes + 128MB
        $free = Get-FreeBytes $Destination
        if ($free -lt $required) {
            throw "Not enough space to extract $([IO.Path]::GetFileName($ZipPath)): about $([math]::Round($required/1GB,2)) GB required, $([math]::Round($free/1GB,2)) GB free. No archive entries were extracted. Free space/use Kaggle storage, then rerun."
        }
        Write-Host "Resuming $([IO.Path]::GetFileName($ZipPath)): $($todo.Count) files remain."
        foreach ($item in $todo) {
            $parent = Split-Path -Parent $item.Target
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
            $temp = $item.Target + '.extracting-' + [guid]::NewGuid().ToString('N')
            try {
                $input = $item.Entry.Open()
                $output = [IO.File]::Open($temp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose() }
                $moved = $false
                for ($attempt = 1; $attempt -le 5 -and -not $moved; $attempt++) {
                    try {
                        Move-Item -LiteralPath $temp -Destination $item.Target -Force -ErrorAction Stop
                        $moved = $true
                    } catch {
                        if ($attempt -eq 5) { throw }
                        Start-Sleep -Milliseconds (250 * $attempt)
                    }
                }
            } catch {
                if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
                throw "Extraction stopped at $($item.Entry.FullName); completed files were preserved. $($_.Exception.Message)"
            }
        }
        Write-Host "Extraction complete: $Destination"
    } finally { $zip.Dispose() }
}

$Kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
foreach ($id in @('kmljts/reside-6k','balraj98/synthetic-objective-testing-set-sots-reside')) {
    $slug = ($id -split '/')[-1]
    $archive = Join-Path $Downloads ($slug + '.zip')
    if (Test-Path -LiteralPath $archive -PathType Leaf) { Write-Host "Keeping archive: $archive"; continue }
    if (-not $Kaggle) { throw 'Kaggle CLI is missing. Install/authenticate it, then rerun. Existing files are preserved.' }
    & $Kaggle.Source datasets download -d $id -p $Downloads
    if ($LASTEXITCODE -ne 0) { throw "Kaggle download failed for $id; existing files are preserved." }
}

Get-ChildItem -LiteralPath $Downloads -Filter '*.zip' -File | Sort-Object Name | ForEach-Object {
    $zipFile = $_
    # Archives can be present but truncated (for example, an interrupted download).
    # Validate each archive independently so one bad ZIP does not block the others.
    $validZip = $false
    try {
        $probe = [IO.Compression.ZipFile]::OpenRead($zipFile.FullName)
        $probe.Dispose()
        $validZip = $true
    } catch {
        Write-Warning "Skipping invalid/incomplete ZIP; preserved for recovery: $($zipFile.FullName)"
    }
    if ($validZip) {
        Expand-ZipResumable -ZipPath $zipFile.FullName -Destination (Join-Path $DataRoot $zipFile.BaseName)
    }
}

$nhZip = Join-Path $Downloads 'NH-HAZE.zip'
$nhValid = $false
if ((Test-Path -LiteralPath $nhZip -PathType Leaf) -and ((Get-Item -LiteralPath $nhZip).Length -gt 0)) {
    try { $test = [IO.Compression.ZipFile]::OpenRead($nhZip); $test.Dispose(); $nhValid = $true } catch { Write-Warning 'Existing NH-HAZE ZIP is incomplete; preserved for recovery.' }
}
if (-not $nhValid) {
    if ((Get-FreeBytes $Downloads) -lt 1GB) {
        Write-Warning 'Skipping NH-HAZE download: less than 1 GB is free on D:. Free space and rerun.'
    } else {
        $partial = Join-Path $Downloads 'NH-HAZE.download.partial'
        try {
            Invoke-WebRequest -Uri 'https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip' -OutFile $partial
            $test = [IO.Compression.ZipFile]::OpenRead($partial); $test.Dispose()
            Move-Item -LiteralPath $partial -Destination $nhZip -Force
            Expand-ZipResumable -ZipPath $nhZip -Destination (Join-Path $DataRoot 'NH-HAZE')
        } catch { Write-Warning "NH-HAZE download did not finish; temporary file preserved at $partial. $_" }
    }
}
Write-Host "Review data destination: $DataRoot"
Write-Host 'Add official O-HAZE, I-HAZE, DNH-HAZE and RTTS ZIPs under downloads, then rerun after freeing enough disk space.'
