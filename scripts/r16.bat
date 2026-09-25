@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "FAILED=0"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BATCH_START_MS=%%A"

echo ==========================================================
echo R16 Induced iTransformer Batch
echo Training from scratch
echo Order: PEMS07, Traffic, ECL, Solar
echo Batch start: %BATCH_START%
echo ==========================================================

REM ==========================================================
REM 1. PEMS07 96-to-96, R16
REM ==========================================================
call :RUN_PEMS ^
    "PEMS07_96_96_induced_r16_tt0_scratch" ^
    96 4 16 0.001

if errorlevel 1 (
    set "FAILED=1"
    goto :BATCH_END
)

REM ==========================================================
REM 2. Traffic 96-to-96, R16
REM ==========================================================
call :RUN_CUSTOM ^
    "traffic_96_96_induced_r16_tt4_scratch" ^
    "./dataset/traffic/" ^
    "traffic.csv" ^
    862 96 4 16 0.001 4

if errorlevel 1 (
    set "FAILED=1"
    goto :BATCH_END
)

REM ==========================================================
REM 3. ECL 96-to-96, R16
REM ==========================================================
call :RUN_CUSTOM ^
    "ECL_96_96_induced_r16_tt4_scratch" ^
    "./dataset/electricity/" ^
    "electricity.csv" ^
    321 96 3 32 0.0005 4

if errorlevel 1 (
    set "FAILED=1"
    goto :BATCH_END
)

REM ==========================================================
REM 4. Solar 96-to-96, R16
REM ==========================================================
call :RUN_SOLAR ^
    "Solar_96_96_induced_r16_tt0_scratch" ^
    96 2 16 0.0005

if errorlevel 1 set "FAILED=1"

goto :BATCH_END


REM ==========================================================
REM Standard custom dataset experiment
REM Args:
REM 1 Model ID
REM 2 Root
REM 3 Data file
REM 4 Channels
REM 5 Prediction length
REM 6 Encoder layers
REM 7 Batch size
REM 8 Learning rate
REM 9 Time tokens
REM ==========================================================
:RUN_CUSTOM
call :EXP_START "%~1"

echo Dataset        : %~2%~3
echo Prediction     : %~5
echo Variables      : %~4
echo Encoder layers : %~6
echo Batch size     : %~7
echo Rank           : 16
echo Time tokens    : %~9
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%~1" ^
 --model iTransformer ^
 --data custom ^
 --root_path "%~2" ^
 --data_path "%~3" ^
 --features M ^
 --target OT ^
 --freq h ^
 --seq_len 96 ^
 --label_len 48 ^
 --pred_len %~5 ^
 --enc_in %~4 ^
 --dec_in %~4 ^
 --c_out %~4 ^
 --d_model 512 ^
 --n_heads 8 ^
 --e_layers %~6 ^
 --d_layers 1 ^
 --d_ff 512 ^
 --factor 1 ^
 --dropout 0.1 ^
 --embed timeF ^
 --batch_size %~7 ^
 --learning_rate %~8 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp ^
 --induced_attention ^
 --attn_rank 16 ^
 --attn_time_tokens %~9 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RC=%ERRORLEVEL%"
call :EXP_END "%RC%"
exit /b %RC%


REM ==========================================================
REM PEMS07 R16
REM Args:
REM 1 Model ID
REM 2 Prediction length
REM 3 Encoder layers
REM 4 Batch size
REM 5 Learning rate
REM ==========================================================
:RUN_PEMS
call :EXP_START "%~1"

echo Dataset        : PEMS07
echo Prediction     : %~2
echo Variables      : 883
echo Encoder layers : %~3
echo Batch size     : %~4
echo Rank           : 16
echo Time tokens    : 0
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%~1" ^
 --model iTransformer ^
 --data PEMS ^
 --root_path ./dataset/PEMS/ ^
 --data_path PEMS07.npz ^
 --features M ^
 --seq_len 96 ^
 --label_len 48 ^
 --pred_len %~2 ^
 --enc_in 883 ^
 --dec_in 883 ^
 --c_out 883 ^
 --d_model 512 ^
 --n_heads 8 ^
 --e_layers %~3 ^
 --d_layers 1 ^
 --d_ff 512 ^
 --factor 1 ^
 --dropout 0.1 ^
 --embed timeF ^
 --batch_size %~4 ^
 --learning_rate %~5 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp ^
 --use_norm 0 ^
 --induced_attention ^
 --attn_rank 16 ^
 --attn_time_tokens 0 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RC=%ERRORLEVEL%"
call :EXP_END "%RC%"
exit /b %RC%


REM ==========================================================
REM Solar R16
REM ==========================================================
:RUN_SOLAR
call :EXP_START "%~1"

echo Dataset        : Solar
echo Prediction     : %~2
echo Variables      : 137
echo Encoder layers : %~3
echo Batch size     : %~4
echo Rank           : 16
echo Time tokens    : 0
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%~1" ^
 --model iTransformer ^
 --data Solar ^
 --root_path ./dataset/Solar/ ^
 --data_path solar_AL.txt ^
 --features M ^
 --seq_len 96 ^
 --label_len 48 ^
 --pred_len %~2 ^
 --enc_in 137 ^
 --dec_in 137 ^
 --c_out 137 ^
 --d_model 512 ^
 --n_heads 8 ^
 --e_layers %~3 ^
 --d_layers 1 ^
 --d_ff 512 ^
 --factor 1 ^
 --dropout 0.1 ^
 --embed timeF ^
 --batch_size %~4 ^
 --learning_rate %~5 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp ^
 --induced_attention ^
 --attn_rank 16 ^
 --attn_time_tokens 0 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RC=%ERRORLEVEL%"
call :EXP_END "%RC%"
exit /b %RC%


:EXP_START
set "CURRENT_MODEL=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_START_MS=%%A"

echo.
echo ==========================================================
echo EXPERIMENT START
echo ==========================================================
echo Model ID   : %CURRENT_MODEL%
echo Start time : %EXP_START%
exit /b 0


:EXP_END
set "EXP_RC=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - %EXP_START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

if "%EXP_RC%"=="0" (
    set "EXP_STATUS=COMPLETED"
) else (
    set "EXP_STATUS=FAILED"
)

echo.
echo ==========================================================
echo EXPERIMENT END
echo ==========================================================
echo Model ID : %CURRENT_MODEL%
echo Status   : %EXP_STATUS%
echo Start    : %EXP_START%
echo End      : %EXP_END%
echo Duration : %EXP_DURATION%
echo Exit code: %EXP_RC%
echo ==========================================================
exit /b 0


:BATCH_END
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - %BATCH_START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "BATCH_DURATION=%%A"

if "%FAILED%"=="0" (
    set "BATCH_STATUS=COMPLETED"
) else (
    set "BATCH_STATUS=FAILED"
)

echo.
echo ==========================================================
echo BATCH SUMMARY
echo ==========================================================
echo Status         : %BATCH_STATUS%
echo Batch start    : %BATCH_START%
echo Batch end      : %BATCH_END%
echo Total duration : %BATCH_DURATION%
echo ==========================================================

popd
pause
exit /b %FAILED%