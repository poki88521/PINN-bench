@echo off
rem run_std.bat - train standard version (baseline)
rem usage: scripts\run_std.bat [-n example_name] [other args]
rem default example: Helmholtz2D ; version fixed: standard
rem example: scripts\run_std.bat -n Wave1D
call scripts\env.bat
%PYTHON% src\main.py -n Helmholtz2D -v standard %*
