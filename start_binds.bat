@echo off
cd /d "%~dp0"
wscript //nologo "%~dp0launch.vbs" skill_bind_tool.py
exit /b
