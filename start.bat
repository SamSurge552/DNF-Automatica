@echo off
cd /d "%~dp0"
wscript //nologo "%~dp0launch.vbs" gui_module.py
exit /b
