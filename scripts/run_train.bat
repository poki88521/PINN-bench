@echo off
rem 训练入口脚本：scripts\run_train.bat -n 算例名 -v 版本名
rem 示例：scripts\run_train.bat -n Helmholtz2D -v ipinn
setlocal
cd /d "%~dp0\.."
set PYTHONPATH=src
D:\anaconda\envs\pinnbench\python.exe src\main.py %*
endlocal
