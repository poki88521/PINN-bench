@echo off
rem run_ipinn.bat - train ipinn version (with control group)
rem usage: scripts\run_ipinn.bat [-n example_name] [other args]
rem default example: Helmholtz2D ; version fixed: ipinn
rem example: scripts\run_ipinn.bat -n Wave1D
call scripts\env.bat
%PYTHON% src\main.py -n Helmholtz2D -v ipinn %*
