<#
.SYNOPSIS
Start the project using its own .venv, with hidden windows and local logs.
.DESCRIPTION
teacher starts the API (8000) and grading workspace (8501); chat starts course Q&A
(8502). Existing recognizable project processes are reused. Other listeners cause
an error; this script never kills processes or installs packages.
.EXAMPLE
.\scripts\start_local.ps1 -Mode teacher
#>
[CmdletBinding()]
param([ValidateSet('teacher', 'chat')][string]$Mode = 'teacher')

function ConvertTo-CommandArgumentPattern {
    param([string]$Value)
    $escaped = [regex]::Escape($Value)
    if ($Value -match '\s') { return ('"' + $escaped + '"(?=\s|$)') }
    return ('(?:"' + $escaped + '"|' + $escaped + ')(?=\s|$)')
}

function Test-ServiceArguments {
    param($Process, $Service)
    if ($null -eq $Process -or [string]::IsNullOrEmpty($Process.CommandLine)) { return $false }
    $command = $Process.CommandLine.Replace('/', '\')
    $entryPattern = ConvertTo-CommandArgumentPattern $Service.EntryPath
    if ($Service.Role -eq 'backend') {
        return $command -match '(?i)\s-m\s+uvicorn\s+backend\.main:app(\s|$)' -and
            $command -match ('(?i)--port\s+' + $Service.Port + '(\s|$)') -and
            $command -match ('(?i)--app-dir\s+' + $entryPattern)
    }
    return $command -match ('(?i)\s-m\s+streamlit\s+run\s+' + $entryPattern) -and
        $command -match ('(?i)--server\.port\s+' + $Service.Port + '(\s|$)')
}

function Test-ProjectProcess {
    param($Process, $Service, [string]$PythonPath, [array]$Processes)
    if (-not (Test-ServiceArguments $Process $Service)) { return $false }
    $command = $Process.CommandLine.Replace('/', '\')
    $pythonPattern = '(?i)^\s*' + (ConvertTo-CommandArgumentPattern $PythonPath)
    if ($Process.ExecutablePath -ieq $PythonPath -or $command -match $pythonPattern) { return $true }

    # Windows venv's launcher can spawn the base interpreter, whose command line
    # also names that base interpreter. Accept only its immediate venv parent,
    # with both processes independently matching this exact service.
    if ($null -eq $Process.ParentProcessId -or $Process.ParentProcessId -eq $Process.ProcessId) { return $false }
    if ([IO.Path]::GetFileName($Process.ExecutablePath) -notin @('python.exe', 'pythonw.exe')) { return $false }
    $childPythonPattern = '(?i)^\s*' + (ConvertTo-CommandArgumentPattern $Process.ExecutablePath)
    if ($command -notmatch $childPythonPattern) { return $false }
    $parents = @($Processes | Where-Object { $_.ProcessId -eq $Process.ParentProcessId })
    if ($parents.Count -ne 1) { return $false }
    $parent = $parents[0]
    if ($parent.ExecutablePath -ine $PythonPath -or [string]::IsNullOrEmpty($parent.CommandLine)) { return $false }
    if ($parent.CommandLine.Replace('/', '\') -notmatch $pythonPattern) { return $false }
    return Test-ServiceArguments $parent $Service
}

function Get-ProjectServicePlan {
    param($Service, [string]$PythonPath, [array]$Connections, [array]$Processes)
    $listeners = @($Connections | Where-Object { $_.LocalPort -eq $Service.Port })
    if ($listeners.Count -gt 0) {
        foreach ($connection in $listeners) {
            $owner = $Processes | Where-Object { $_.ProcessId -eq $connection.OwningProcess } | Select-Object -First 1
            if (-not (Test-ProjectProcess -Process $owner -Service $Service -PythonPath $PythonPath -Processes $Processes)) {
                throw "Port $($Service.Port) is occupied by another process (PID $($connection.OwningProcess)). Nothing will be stopped."
            }
        }
        return 'reuse'
    }
    foreach ($process in $Processes) {
        if (Test-ProjectProcess -Process $process -Service $Service -PythonPath $PythonPath -Processes $Processes) {
            return 'starting'
        }
    }
    return 'start'
}

function Start-LocalProject {
    param([string]$SelectedMode)
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw 'Project .venv is missing. In the project folder, run: py -3.12 -m venv .venv; then .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    }
    $services = @()
    if ($SelectedMode -eq 'teacher') {
        $services += [pscustomobject]@{
            Role = 'backend'; Port = 8000; Url = 'http://127.0.0.1:8000/'; EntryPath = $projectRoot
            Arguments = @('-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000', '--app-dir', ('"{0}"' -f $projectRoot))
        }
        $entryPath = Join-Path $projectRoot 'frontend\app.py'
        $port = 8501
    } else {
        $entryPath = Join-Path $projectRoot 'app.py'
        $port = 8502
    }
    $services += [pscustomobject]@{
        Role = $SelectedMode; Port = $port; Url = "http://127.0.0.1:$port/"; EntryPath = $entryPath
        Arguments = @('-m', 'streamlit', 'run', ('"{0}"' -f $entryPath), '--server.address', '127.0.0.1', '--server.port', "$port", '--server.headless', 'true')
    }

    # Serialize concurrent clicks. The lock only exists during this short startup.
    $digest = [System.Security.Cryptography.SHA256]::Create()
    try { $rootHash = [BitConverter]::ToString($digest.ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant()))).Replace('-', '') }
    finally { $digest.Dispose() }
    $mutex = [Threading.Mutex]::new($false, "Local\AutocontrolStart$rootHash")
    $locked = $false
    try {
        try { $locked = $mutex.WaitOne(0) }
        catch [Threading.AbandonedMutexException] { $locked = $true }
        if (-not $locked) { throw 'Another project startup is already in progress. Please wait for it to finish.' }
        $connections = @(Get-NetTCPConnection -State Listen -ErrorAction Stop)
        $processes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop)
        $plans = @()
        # Preflight every target before starting any service.
        foreach ($service in $services) {
            $plans += Get-ProjectServicePlan -Service $service -PythonPath $pythonPath -Connections $connections -Processes $processes
        }
        $logDirectory = Join-Path $projectRoot 'output\runtime'
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        for ($index = 0; $index -lt $services.Count; $index++) {
            $service = $services[$index]
            if ($plans[$index] -eq 'start') {
                $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
                $stdout = Join-Path $logDirectory "$($service.Role)-$timestamp.out.log"
                $stderr = Join-Path $logDirectory "$($service.Role)-$timestamp.err.log"
                $started = Start-Process -FilePath $pythonPath -ArgumentList $service.Arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
                Write-Output "Started $($service.Role) (PID $($started.Id)): $($service.Url)"
                Write-Output "Startup may take a moment. Logs: $stdout | $stderr"
            } elseif ($plans[$index] -eq 'reuse') {
                Write-Output "Already running $($service.Role): $($service.Url)"
            } else {
                Write-Output "$($service.Role) is already starting: $($service.Url) (check output/runtime logs if it does not become ready)."
            }
        }
    } finally {
        if ($locked) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    try { Start-LocalProject -SelectedMode $Mode }
    catch { Write-Error $_; exit 1 }
}
