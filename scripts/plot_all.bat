@echo off
rem plot_all.bat - draw plots for all examples of one version
rem usage: scripts\plot_all.bat [version]   (default: standard)
rem examples:
rem   scripts\plot_all.bat
rem   scripts\plot_all.bat ipinn
call "%~dp0env.bat"
set "VERSION=standard"
if not "%~1"=="" set "VERSION=%~1"
for %%E in (Heat1D Helmholtz2D Wave1D AllenCahn1D Burgers1D) do (
    echo ===== Plot %%E ^(version: %VERSION%^) =====
    %PYTHON% "%PROJECT_ROOT%\src\plot.py" -n %%E -v %VERSION%
)
echo ===== All done =====
pause
