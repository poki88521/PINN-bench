@echo off
rem run_all.bat - train all examples for one version
rem usage: scripts\run_all.bat [version]   (default: standard)
rem examples:
rem   scripts\run_all.bat
rem   scripts\run_all.bat ipinn
call "%~dp0env.bat"
set "VERSION=standard"
if not "%~1"=="" set "VERSION=%~1"
for %%E in (Heat1D Helmholtz2D Wave1D AllenCahn1D Burgers1D) do (
    echo ===== Run %%E ^(version: %VERSION%^) =====
    %PYTHON% "%PROJECT_ROOT%\src\main.py" -n %%E -v %VERSION%
)
echo ===== All done =====
pause
