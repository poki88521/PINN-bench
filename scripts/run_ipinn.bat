@echo off
rem run_ipinn.bat - train ipinn version (with control group)
rem usage: scripts\run_ipinn.bat [-n example_name] [other args]
rem default example: Helmholtz2D ; version fixed: ipinn
rem example: scripts\run_ipinn.bat -n Wave1D
call "%~dp0env.bat"
%PYTHON% "%PROJECT_ROOT%\src\main.py" -n Helmholtz2D -v ipinn %*
pause
