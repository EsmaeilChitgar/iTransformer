@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "WARMUP=30"
set "ITERATIONS=100"
set "TRAINING_WARMUP=10"
set "TRAINING_ITERATIONS=30"
set "DTYPE=float32"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "RUN_ID=%%A"
set "OUTPUT_DIR=./benchmark_results/paper_minimal_%RUN_ID%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

call "%PYTHON%" -c "import torch,sys; print('PyTorch:',torch.__version__); print('CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 goto :FAIL

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BATCH_START_MS=%%A"
set /a PASSED=0
set /a FAILED=0

echo ==========================================================
echo Full-model Paper Benchmark - Dense versus Low-rank
echo No fitting and no dataset access
echo Inference latency, throughput, and forward/backward timing
echo Output: %OUTPUT_DIR%
echo Start:  %BATCH_START%
echo ==========================================================

(
  echo Run ID: %RUN_ID%
  echo Start: %BATCH_START%
  git rev-parse HEAD 2^>nul
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2^>nul
  call "%PYTHON%" -c "import torch; print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda))"
) > "%OUTPUT_DIR%\RUN_INFO.txt" 2>&1

REM Traffic 96: the complete rank/accuracy scaling curve.
call :RUN traffic_96_dense dense 16 0 96 862 4 4 16
call :RUN traffic_96_r8_tt4 induced 8 4 96 862 4 4 16
call :RUN traffic_96_r16_tt4 induced 16 4 96 862 4 4 16
call :RUN traffic_96_r32_tt4 induced 32 4 96 862 4 4 16
call :RUN traffic_96_r64_tt4 induced 64 4 96 862 4 4 16

REM Traffic 720: long-horizon end-to-end projection cost.
call :RUN traffic_720_dense dense 16 0 720 862 4 4 16
call :RUN traffic_720_r16_tt4 induced 16 4 720 862 4 4 16
call :RUN traffic_720_r64_tt4 induced 64 4 720 862 4 4 16

REM PEMS03 48: independent high-dimensional dataset, no time tokens.
call :RUN pems03_48_dense dense 16 0 48 358 0 4 32
call :RUN pems03_48_r16_tt0 induced 16 0 48 358 0 4 32
call :RUN pems03_48_r32_tt0 induced 32 0 48 358 0 4 32

REM ECL 96: small-token boundary case, included for an honest comparison.
call :RUN ecl_96_dense dense 16 0 96 321 4 3 32
call :RUN ecl_96_r16_tt4 induced 16 4 96 321 4 3 32

goto :SUMMARY

:RUN
set "EXP_NAME=%~1"
set "ATTENTION=%~2"
set "RANK=%~3"
set "TIME_TOKENS=%~4"
set "PRED_LEN=%~5"
set "ENC_IN=%~6"
set "TIME_FEATURES=%~7"
set "E_LAYERS=%~8"
set "THROUGHPUT_BATCH=%~9"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_START_MS=%%A"

echo.
echo ==========================================================
echo START: !EXP_NAME!
echo Attention: !ATTENTION!  Rank: !RANK!  Time tokens: !TIME_TOKENS!
echo Start time: !EXP_START!
echo ==========================================================

call "%PYTHON%" -u tests\benchmark_forecasting_model.py ^
  --name "!EXP_NAME!" ^
  --output_dir "%OUTPUT_DIR%" ^
  --attention "!ATTENTION!" ^
  --rank !RANK! ^
  --time_tokens !TIME_TOKENS! ^
  --seq_len 96 ^
  --pred_len !PRED_LEN! ^
  --enc_in !ENC_IN! ^
  --time_features !TIME_FEATURES! ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers !E_LAYERS! ^
  --d_ff 512 ^
  --use_norm 1 ^
  --latency_batch_size 1 ^
  --throughput_batch_size !THROUGHPUT_BATCH! ^
  --warmup %WARMUP% ^
  --iterations %ITERATIONS% ^
  --training_warmup %TRAINING_WARMUP% ^
  --training_iterations %TRAINING_ITERATIONS% ^
  --device cuda ^
  --dtype %DTYPE% > "%OUTPUT_DIR%\!EXP_NAME!.log" 2>&1
set "EXP_CODE=!ERRORLEVEL!"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - !EXP_START_MS!)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

if !EXP_CODE! equ 0 (
  set /a PASSED+=1
  echo END: !EXP_NAME! - COMPLETED - !EXP_DURATION!
) else (
  set /a FAILED+=1
  echo END: !EXP_NAME! - FAILED !EXP_CODE! - !EXP_DURATION!
  type "%OUTPUT_DIR%\!EXP_NAME!.log"
)
exit /b 0

:SUMMARY
if exist "%OUTPUT_DIR%\benchmark_summary.csv" (
  call "%PYTHON%" tests\summarize_paper_benchmark.py "%OUTPUT_DIR%\benchmark_summary.csv" > "%OUTPUT_DIR%\benchmark_comparison.txt" 2>&1
  type "%OUTPUT_DIR%\benchmark_comparison.txt"
)
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - %BATCH_START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "BATCH_DURATION=%%A"

echo.
echo ==========================================================
echo BENCHMARK SUMMARY
echo Passed: !PASSED!  Failed: !FAILED!
echo Start: %BATCH_START%
echo End:   !BATCH_END!
echo Total: !BATCH_DURATION!
echo CSV:   %OUTPUT_DIR%\benchmark_summary.csv
echo ==========================================================
popd
pause
if !FAILED! gtr 0 exit /b 1
exit /b 0

:FAIL
echo ERROR: CUDA PyTorch is required.
popd
pause
exit /b 1
