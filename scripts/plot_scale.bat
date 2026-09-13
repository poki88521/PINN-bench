@echo off
rem plot_scale.bat - draw plots for scale version (Scale-PINN)
rem usage: scripts\plot_scale.bat [-n example_name] [other args]
rem default example: AllenCahn1D (currently the only implemented example) ; version fixed: scale
rem draws three pngs (history / solution / solution slice) and prints RL2 against ipinn
rem field plots use <base>_best.pt when it exists, otherwise the last checkpoint
rem all extra args are passed to src\plot_scale.py, examples:
rem   scripts\plot_scale.bat                          (plots of runs\AllenCahn1D\scale)
rem   scripts\plot_scale.bat --tag ER0                (same --tag as run_scale.bat)
rem   scripts\plot_scale.bat --ipinn_dir D:\some\where (ipinn runs dir; missing dir only warns)
call "%~dp0env.bat"
%PYTHON% "%PROJECT_ROOT%\src\plot_scale.py" -n AllenCahn1D -v scale %*
pause
