param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsForDailyStonks
)

$ErrorActionPreference = "Stop"
$EngineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = $EngineRoot.Path
python -m dailystonks.live_terminal @ArgsForDailyStonks
