Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
Set app = CreateObject("Shell.Application")

folder = fso.GetParentFolderName(WScript.ScriptFullName)
py = sh.ExpandEnvironmentStrings("%LocalAppData%\Programs\Python\Python312\pythonw.exe")
If Not fso.FileExists(py) Then py = "pythonw.exe"
scriptName = "gui_module.py"
If WScript.Arguments.Count > 0 Then scriptName = WScript.Arguments(0)
noelevate = False
If WScript.Arguments.Count > 1 Then
  If LCase(WScript.Arguments(1)) = "noelevate" Then noelevate = True
End If
scriptPath = folder & "\" & scriptName

If (Not noelevate) And (Not IsAdmin()) Then
  ' 无控制台：先在普通权限里快照 PATH，再 UAC 打开目标脚本
  app.ShellExecute py, "launch_elevated.py """ & scriptPath & """", folder, "open", 0
Else
  app.ShellExecute py, """" & scriptPath & """", folder, "open", 1
End If

Function IsAdmin()
  On Error Resume Next
  CreateObject("WScript.Shell").RegRead "HKEY_USERS\S-1-5-19\"
  IsAdmin = (Err.Number = 0)
  Err.Clear
End Function
