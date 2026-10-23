@echo off
REM ============================================================
REM  Setup + GPU runner (jalankan dengan dobel klik)
REM  - Memakai Python yang sudah ada (yang punya PyTorch CUDA), tanpa venv
REM  - Memasang library proyek yang belum ada, lalu menjalankan GPU runner
REM ============================================================
cd /d "%~dp0"
if not exist jobs mkdir jobs
set LOG=jobs\setup.log
echo [%date% %time%] mulai setup > %LOG%

set PY=
for %%P in (python "py -3") do (
  if not defined PY (
    %%~P -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && set "PY=%%~P"
  )
)
if not defined PY (
  echo PERINGATAN: tidak menemukan Python dengan PyTorch CUDA. Memakai "python".
  echo TIDAK ADA torch CUDA >> %LOG%
  set "PY=python"
)
echo Python dipakai: %PY%
echo PY=%PY% >> %LOG%
%PY% -c "import sys,torch;print(sys.executable);print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())" >> %LOG% 2>&1

if exist jobs\.setup_done goto :run
echo Memasang library proyek yang belum ada...
%PY% -m pip install -r requirements.txt >> %LOG% 2>&1
if errorlevel 1 (
  echo GAGAL memasang library. Lihat jobs\setup.log
  pause
  exit /b 1
)
echo ok > jobs\.setup_done
echo [%date% %time%] setup selesai >> %LOG%

:run
echo.
echo Menjalankan GPU runner. Biarkan jendela ini terbuka.
%PY% runner\gpu_runner.py
pause
