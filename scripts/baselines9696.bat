@echo off
setlocal EnableExtensions

pushd "%~dp0.."
if errorlevel 1 (
    echo ERROR: Cannot enter repository root.
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

if not exist "./dataset/electricity/electricity.csv" (
    echo ERROR: ECL dataset not found.
    goto :INITIAL_FAILURE
)

if not exist "./dataset/weather/weather.csv" (
    echo ERROR: Weather dataset not found.
    goto :INITIAL_FAILURE
)

if not exist "./dataset/Solar/solar_AL.txt" (
    echo ERROR: Solar dataset not found.
    goto :INITIAL_FAILURE
)

call "%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is unavailable.
    goto :INITIAL_FAILURE
)

call :BATCH_START

echo.
echo ==========================================================
echo Dense Baseline Batch
echo Experiments: ECL-96, Weather-96, Solar-96
echo Batch start: %BATCH_START_TIME%
echo ==========================================================

call :RUN_ECL
if errorlevel 1 goto :BATCH_FAILED

call :RUN_WEATHER
if errorlevel 1 goto :BATCH_FAILED

call :RUN_SOLAR
if errorlevel 1 goto :BATCH_FAILED

set "BATCH_STATUS=COMPLETED"
set "BATCH_EXIT_CODE=0"
goto :BATCH_FINISH

:BATCH_FAILED
set "BATCH_STATUS=FAILED"
set "BATCH_EXIT_CODE=1"
goto :BATCH_FINISH

:INITIAL_FAILURE
set "BATCH_STATUS=FAILED"
set "BATCH_EXIT_CODE=1"
popd
pause
exit /b 1

:BATCH_FINISH
call :BATCH_END

echo.
echo ==========================================================
echo BATCH SUMMARY
echo ==========================================================
echo Status         : %BATCH_STATUS%
echo Batch start    : %BATCH_START_TIME%
echo Batch end      : %BATCH_END_TIME%
echo Total duration : %BATCH_DURATION%
echo Exit code      : %BATCH_EXIT_CODE%
echo ==========================================================

popd
pause
exit /b %BATCH_EXIT_CODE%


:RUN_ECL
set "EXP_NAME=ECL_96_96_dense_control"
call :EXP_START

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%EXP_NAME%" ^
 --model iTransformer ^
 --data custom ^
 --root_path ./dataset/electricity/ ^
 --data_path electricity.csv ^
 --features M ^
 --seq_len 96 ^
 --pred_len 96 ^
 --e_layers 3 ^
 --enc_in 321 ^
 --dec_in 321 ^
 --c_out 321 ^
 --d_model 512 ^
 --d_ff 512 ^
 --batch_size 32 ^
 --learning_rate 0.0005 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp

set "RUN_CODE=%ERRORLEVEL%"
call :EXP_END %RUN_CODE%
exit /b %RUN_CODE%


:RUN_WEATHER
set "EXP_NAME=weather_96_96_dense_control"
call :EXP_START

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%EXP_NAME%" ^
 --model iTransformer ^
 --data custom ^
 --root_path ./dataset/weather/ ^
 --data_path weather.csv ^
 --features M ^
 --seq_len 96 ^
 --pred_len 96 ^
 --e_layers 3 ^
 --enc_in 21 ^
 --dec_in 21 ^
 --c_out 21 ^
 --d_model 512 ^
 --d_ff 512 ^
 --batch_size 32 ^
 --learning_rate 0.0001 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp

set "RUN_CODE=%ERRORLEVEL%"
call :EXP_END %RUN_CODE%
exit /b %RUN_CODE%


:RUN_SOLAR
set "EXP_NAME=solar_96_96_dense_control"
call :EXP_START

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%EXP_NAME%" ^
 --model iTransformer ^
 --data Solar ^
 --root_path ./dataset/Solar/ ^
 --data_path solar_AL.txt ^
 --features M ^
 --seq_len 96 ^
 --pred_len 96 ^
 --e_layers 2 ^
 --enc_in 137 ^
 --dec_in 137 ^
 --c_out 137 ^
 --d_model 512 ^
 --d_ff 512 ^
 --batch_size 16 ^
 --learning_rate 0.0005 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp

set "RUN_CODE=%ERRORLEVEL%"
call :EXP_END %RUN_CODE%
exit /b %RUN_CODE%


:EXP_START
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_START_MS=%%A"

echo.
echo ==========================================================
echo EXPERIMENT START
echo Model ID   : %EXP_NAME%
echo Start time : %EXP_START_TIME%
echo ==========================================================
exit /b 0


:EXP_END
set "RUN_CODE=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_END_MS=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=(%EXP_END_MS%-%EXP_START_MS%)/1000; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=[math]::Floor($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

if "%RUN_CODE%"=="0" (
    set "EXP_STATUS=COMPLETED"
) else (
    set "EXP_STATUS=FAILED"
)

echo.
echo ==========================================================
echo EXPERIMENT END
echo Model ID : %EXP_NAME%
echo Status   : %EXP_STATUS%
echo Start    : %EXP_START_TIME%
echo End      : %EXP_END_TIME%
echo Duration : %EXP_DURATION%
echo Exit code: %RUN_CODE%
echo ==========================================================
exit /b 0


:BATCH_START
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_START_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BATCH_START_MS=%%A"
exit /b 0


:BATCH_END
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_END_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BATCH_END_MS=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=(%BATCH_END_MS%-%BATCH_START_MS%)/1000; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=[math]::Floor($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "BATCH_DURATION=%%A"
exit /b 0