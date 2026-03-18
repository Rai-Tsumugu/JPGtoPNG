$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" convert.py @args
} else {
    python convert.py @args
}
