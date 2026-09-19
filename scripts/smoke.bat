@echo off
rem smoke.bat - quick smoke test for scale version (100 iterations, ~15 s)
rem usage:
rem   scripts\smoke.bat                 -> ER=0.45, seed 0, tag "smoke"
rem   scripts\smoke.bat [args]          -> extra args are appended, e.g. --ER 0
rem artifacts go to runs\AllenCahn1D\scale\smoke\ (delete the folder by hand when done)
rem real experiments: scripts\run_scale.bat (no args)
call "%~dp0env.bat"

%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --iterations 100 --display_every 10 --tag smoke %*
pause
