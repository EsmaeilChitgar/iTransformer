@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Run from the repository root
pushd "%~dp0.."
if errorlevel 1 (
    echo ERROR: Cannot enter repository root.
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

set "DATA_ROOT=./dataset/Solar/"
set "DATA_FILE=solar_AL.txt"

if not exist "%DATA_ROOT%%DATA_FILE%" (
    echo ERROR: Dataset not found:
    echo %DATA_ROOT%%DATA_FILE%
    pause
    popd
    exit /b 1
)

REM ==========================================================
REM Record total start time
REM ==========================================================

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "TOTAL_START_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "TOTAL_START_SECONDS=%%A"

set "BATCH_EXIT_CODE=0"

REM ==========================================================
REM Experiment 1
REM Solar 96-to-96, Induced Attention R64, From Scratch
REM ==========================================================

call :EXP_START "solar_96_96_induced_r64_scratch_seed2023"

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id solar_96_96_induced_r64_scratch_seed2023 ^
 --model iTransformer ^
 --root_path "%DATA_ROOT%" ^
 --data_path "%DATA_FILE%" ^
 --data Solar ^
 --features M ^
 --seq_len 96 ^
 --pred_len 96 ^
 --e_layers 2 ^
 --enc_in 137 ^
 --dec_in 137 ^
 --c_out 137 ^
 --des Exp ^
 --d_model 512 ^
 --d_ff 512 ^
 --batch_size 16 ^
 --learning_rate 0.0005 ^
 --num_workers 0 ^
 --itr 1 ^
 --induced_attention ^
 --attn_rank 64 ^
 --attn_time_tokens 0 ^
 --attn_gate_init 1.0

if errorlevel 1 (
    set "BATCH_EXIT_CODE=1"
    call :EXP_END "FAILED"
    goto :ALL_DONE
) else (
    call :EXP_END "COMPLETED"
)



REM ==========================================================
REM End of all experiments
REM ==========================================================

:ALL_DONE

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "TOTAL_END_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "TOTAL_END_SECONDS=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=%TOTAL_END_SECONDS%-%TOTAL_START_SECONDS%; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=$d%%60; '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "TOTAL_DURATION=%%A"

echo.
echo ==========================================================
echo BATCH SUMMARY
echo ==========================================================
echo Batch started : %TOTAL_START_TIME%
echo Batch ended   : %TOTAL_END_TIME%
echo Total duration: %TOTAL_DURATION%
echo Exit code     : %BATCH_EXIT_CODE%
echo ==========================================================

popd
pause
exit /b %BATCH_EXIT_CODE%

REM ==========================================================
REM Timing helpers
REM ==========================================================

:EXP_START
set "CURRENT_EXP_NAME=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "EXP_START_SECONDS=%%A"

echo.
echo ==========================================================
echo [START] %CURRENT_EXP_NAME%
echo Start time: %EXP_START_TIME%
echo ==========================================================
exit /b 0

:EXP_END
set "EXP_STATUS=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "EXP_END_SECONDS=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=%EXP_END_SECONDS%-%EXP_START_SECONDS%; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=$d%%60; '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

echo.
echo ==========================================================
echo [END - %EXP_STATUS%] %CURRENT_EXP_NAME%
echo Start time: %EXP_START_TIME%
echo End time  : %EXP_END_TIME%
echo Duration  : %EXP_DURATION%
echo ==========================================================
exit /b 0