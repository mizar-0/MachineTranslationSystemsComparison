param([int]$QueueProcessId)

$signature = @"
using System;
using System.Runtime.InteropServices;
public static class ExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
Add-Type -TypeDefinition $signature

$continuous = [uint32]0x80000000
$systemRequired = [uint32]0x00000001
[ExecutionState]::SetThreadExecutionState($continuous -bor $systemRequired) | Out-Null
try {
    while (Get-Process -Id $QueueProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
} finally {
    [ExecutionState]::SetThreadExecutionState($continuous) | Out-Null
}
