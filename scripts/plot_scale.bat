@echo off
rem plot_scale.bat - draw all scale figures in one run
rem usage:
rem   scripts\plot_scale.bat          -> all figures (see below)
rem   scripts\plot_scale.bat [args]   -> single run, args are passed to src\plot_scale.py
rem
rem all figures = 6 runs x 3 pngs + 4 seed-comparison pngs = 22 pngs
rem   per run: <base>_history.png / <base>_solution.png / <base>_solution_slice.png
rem   seeds  : AllenCahn1D_scale_seeds_l2 / _loss / _solution / _slice .png
rem tags below must match scripts\run_scale.bat
rem field plots use <base>_best.pt when it exists, otherwise the last checkpoint
rem
rem single run examples:
rem   scripts\plot_scale.bat --tag ER0_s0
rem   scripts\plot_scale.bat --ipinn_dir D:\some\where   (ipinn runs dir; missing dir only warns)
call "%~dp0env.bat"

rem mode 2: single run passthrough
if not "%~1"=="" (
    %PYTHON% "%PROJECT_ROOT%\src\plot_scale.py" -n AllenCahn1D -v scale %*
    pause
    exit /b
)

rem mode 1: all runs (run scripts\run_scale.bat first)
set "TAGS=ER0_s0 ER0_s10 ER0_s20 ER045_s0 ER045_s10 ER045_s20"

echo ===== per-run plots (3 pngs per run) =====
for %%T in (%TAGS%) do (
    echo --- %%T ---
    %PYTHON% "%PROJECT_ROOT%\src\plot_scale.py" -n AllenCahn1D -v scale --tag %%T
)

echo ===== seed comparison plots (4 pngs) =====
%PYTHON% "%PROJECT_ROOT%\src\plot_seeds.py" -n AllenCahn1D -v scale

echo ===== All plots done =====
pause
