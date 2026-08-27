@echo off
rem ============================================================
rem env.bat - environment bootstrap, called by other scripts via:
rem   call scripts\env.bat
rem sets: PROJECT_ROOT, PYTHON, PYTHONPATH, and cd to project root
rem ============================================================

rem project root = parent of scripts\ dir
set "PROJECT_ROOT=%~dp0.."

rem python interpreter
set "PYTHON=D:\anaconda\envs\pinnbench\python.exe"

rem pythonpath points to src
set "PYTHONPATH=%PROJECT_ROOT%\src"

rem switch to project root
cd /d "%PROJECT_ROOT%"
