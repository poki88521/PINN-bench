@echo off
rem run_scale.bat - train scale version (Scale-PINN, sequential correction)
rem usage: scripts\run_scale.bat [-n example_name] [other args]
rem default example: AllenCahn1D (currently the only implemented example) ; version fixed: scale
rem all extra args are passed to src\main_scale.py, examples:
rem   scripts\run_scale.bat --tag main                 (default config: 40000 iters, ER=0.45)
rem   scripts\run_scale.bat --ER 0 --tag ER0           (official ref control, correction off)
rem   scripts\run_scale.bat --seed 10 --tag seed10
rem   scripts\run_scale.bat --iterations 100 --display_every 10 --tag smoke
rem --tag X writes to runs\<name>\scale\X\ with _X in file names, so runs do not overwrite each other
call "%~dp0env.bat"
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale %*
pause
