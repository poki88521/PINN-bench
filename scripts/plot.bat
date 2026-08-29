@echo off
rem plot.bat - draw plots
rem usage: scripts\plot.bat [-n example_name] [-v version] [-o plot1 plot2 ...]
rem default: Helmholtz2D standard, draw all plots (no -o)
rem examples:
rem   scripts\plot.bat
rem   scripts\plot.bat -v ipinn
rem   scripts\plot.bat -v ipinn -o solution_plot history_compare_plot
call "%~dp0env.bat"
%PYTHON% "%PROJECT_ROOT%\src\plot.py" -n Helmholtz2D -v standard %*
pause
