@echo off
rem plot_seeds.bat - seed comparison figures for the 6 scale runs
rem usage: scripts\plot_seeds.bat [extra args]
rem reads runs\AllenCahn1D\scale\<tag>\ for tags ER045_s0/s10/s20 and ER0_s0/s10/s20
rem writes 4 pngs into runs\AllenCahn1D\scale\:
rem   AllenCahn1D_scale_seeds_l2.png        l2 curves + end-value scatter + paired-by-seed
rem   AllenCahn1D_scale_seeds_loss.png      train / test loss, 6 runs each
rem   AllenCahn1D_scale_seeds_solution.png  |Pred-Exact| heatmaps (2 ER groups x 3 seeds)
rem   AllenCahn1D_scale_seeds_slice.png     5 time slices, 6 predictions + exact
rem run scripts\run_scale.bat (all 6 runs) first; weights prefer the best checkpoints
call "%~dp0env.bat"
%PYTHON% "%PROJECT_ROOT%\src\plot_seeds.py" -n AllenCahn1D -v scale %*
pause
