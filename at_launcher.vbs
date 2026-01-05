Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Use pythonw.exe from PATH (no console window)
' Launch at_launcher.pyw with no window (0 = hidden, 1 = normal, 2 = minimized)
WshShell.Run "pythonw.exe """ & scriptDir & "\at_launcher.pyw""", 0, False

Set WshShell = Nothing
Set fso = Nothing
