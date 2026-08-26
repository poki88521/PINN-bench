@echo off
rem 绘图入口脚本：scripts\run_plot.bat -n 算例名 -v 版本名 [-o 图名1 图名2 ...]
rem 示例：scripts\run_plot.bat -n Helmholtz2D -v ipinn
rem       scripts\run_plot.bat -n Helmholtz2D -v ipinn -o solution_plot history_compare_plot
setlocal
cd /d "%~dp0\.."
set PYTHONPATH=src
D:\anaconda\envs\pinnbench\python.exe src\plot.py %*
endlocal
