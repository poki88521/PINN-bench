@echo off
rem run_scale.bat - train scale version (Scale-PINN, sequential correction)
rem usage:
rem   scripts\run_scale.bat          -> full experiment set (6 runs, see below)
rem   scripts\run_scale.bat [args]   -> single custom run, args passed to src\main_scale.py
rem
rem full set: ER=0.45 and ER=0 (correction off, official ref control), each with seed 0/10/20
rem   tags: ER045_s0 / ER045_s10 / ER045_s20 / ER0_s0 / ER0_s10 / ER0_s20
rem every run: cosine lr schedule with decay_steps = iterations = 40000 (max iter is 40000)
rem
rem single run examples:
rem   scripts\run_scale.bat --iterations 100 --display_every 10 --tag smoke
rem   scripts\run_scale.bat --ER 0 --seed 10 --tag ER0_s10
rem --tag X writes to runs\<name>\scale\X\ with _X in file names, so runs do not overwrite each other
call "%~dp0env.bat"

rem mode 2: single custom run
if not "%~1"=="" (
    %PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale %*
    pause
    exit /b
)

rem mode 1: full experiment set (6 runs in sequence)
echo ===== ER=0.45 (correction on) =====
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0.45 --seed 0 --tag ER045_s0
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0.45 --seed 10 --tag ER045_s10
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0.45 --seed 20 --tag ER045_s20
echo ===== ER=0 (correction off, official ref control) =====
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0 --seed 0 --tag ER0_s0
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0 --seed 10 --tag ER0_s10
%PYTHON% "%PROJECT_ROOT%\src\main_scale.py" -n AllenCahn1D -v scale --ER 0 --seed 20 --tag ER0_s20
echo ===== All 6 runs done =====
pause
