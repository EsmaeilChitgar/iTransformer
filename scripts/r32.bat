@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "RANK=32"
set "FAILED=0"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "BATCH_START_MS=%%A"

echo ==========================================================
echo R32 High-dimensional Induced Attention Batch
echo Training from scratch
echo Experiments: Traffic-96, PEMS07-96
echo Batch start: %BATCH_START%
echo ==========================================================

call :RUN_TRAFFIC
if errorlevel 1 (
    set "FAILED=1"
    goto :BATCH_END
)

call :RUN_PEMS
if errorlevel 1 set "FAILED=1"

goto :BATCH_END


:RUN_TRAFFIC
set "MODEL_ID=traffic_96_96_induced_r32_tt4_scratch"
call :EXP_START

echo Dataset     : Traffic
echo Variables   : 862
echo Prediction  : 96
echo Rank        : %RANK%
echo Time tokens : 4
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%MODEL_ID%" ^
 --model iTransformer ^
 --data custom ^
 --root_path ./dataset/traffic/ ^
 --data_path traffic.csv ^
 --features M ^
 --target OT ^
 --freq h ^
 --seq_len 96 ^
 --label_len 48 ^
 --pred_len 96 ^
 --enc_in 862 ^
 --dec_in 862 ^
 --c_out 862 ^
 --d_model 512 ^
 --n_heads 8 ^
 --e_layers 4 ^
 --d_layers 1 ^
 --d_ff 512 ^
 --factor 1 ^
 --dropout 0.1 ^
 --embed timeF ^
 --batch_size 16 ^
 --learning_rate 0.001 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp ^
 --induced_attention ^
 --attn_rank %RANK% ^
 --attn_time_tokens 4 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RC=%ERRORLEVEL%"
call :EXP_END "%RC%"
exit /b %RC%


:RUN_PEMS
set "MODEL_ID=PEMS07_96_96_induced_r32_tt0_scratch"
call :EXP_START

echo Dataset     : PEMS07
echo Variables   : 883
echo Prediction  : 96
echo Rank        : %RANK%
echo Time tokens : 0
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --model_id "%MODEL_ID%" ^
 --model iTransformer ^
 --data PEMS ^
 --root_path ./dataset/PEMS/ ^
 --data_path PEMS07.npz ^
 --features M ^
 --seq_len 96 ^
 --label_len 48 ^
 --pred_len 96 ^
 --enc_in 883 ^
 --dec_in 883 ^
 --c_out 883 ^
 --d_model 512 ^
 --n_heads 8 ^
 --e_layers 4 ^
 --d_layers 1 ^
 --d_ff 512 ^
 --factor 1 ^
 --dropout 0.1 ^
 --embed timeF ^
 --batch_size 16 ^
 --learning_rate 0.001 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --des Exp ^
 --use_norm 0 ^
 --induced_attention ^
 --attn_rank %RANK% ^
 --attn_time_tokens 0 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RC=%ERRORLEVEL%"
call :EXP_END "%RC%"
exit /b %RC%


:EXP_START
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_START_MS=%%A"

echo.
echo ==========================================================
echo EXPERIMENT START
echo Model ID   : %MODEL_ID%
echo Start time : %EXP_START%
echo ==========================================================
exit /b 0


:EXP_END
set "RC=%~1"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds()-%EXP_START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

if "%RC%"=="0" (
    set "STATUS=COMPLETED"
) else (
    set "STATUS=FAILED"
)

echo.
echo ==========================================================
echo EXPERIMENT END
echo Model ID : %MODEL_ID%
echo Status   : %STATUS%
echo Start    : %EXP_START%
echo End      : %EXP_END%
echo Duration : %EXP_DURATION%
echo Exit code: %RC%
echo ==========================================================
exit /b 0


:BATCH_END
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BATCH_END=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds()-%BATCH_START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "BATCH_DURATION=%%A"

echo.
echo ==========================================================
echo BATCH SUMMARY
echo Status         : !FAILED!
echo Batch start    : %BATCH_START%
echo Batch end      : %BATCH_END%
echo Total duration : %BATCH_DURATION%
echo ==========================================================

popd
pause
exit /b %FAILED%