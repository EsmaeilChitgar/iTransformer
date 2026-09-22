@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM iTransformer Traffic - Global Fixed Low-Rank Attention
REM First viability experiment: R = 64
REM ==========================================================

set CUDA_VISIBLE_DEVICES=0

REM ----------------------------------------------------------
REM Experiment configuration
REM ----------------------------------------------------------

set "RANK=64"
set "ATTN_TOKENS=866"
set "MODEL_ID=traffic_96_96_lr64"

REM Same architecture/settings as the existing Full model.
set "BASELINE_MODEL_ID=traffic_96_96_rank_diag"

REM Existing Full-attention checkpoint.
set "BASELINE_CHECKPOINT=.\checkpoints\traffic_96_96_rank_diag_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0\checkpoint.pth"

REM Expected Low-Rank checkpoint after training.
set "LOWRANK_CHECKPOINT=.\checkpoints\traffic_96_96_lr64_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0\checkpoint.pth"


echo ==========================================================
echo iTransformer Traffic - Global Fixed Low-Rank Attention
echo ==========================================================
echo Rank:
echo %RANK%
echo.
echo Attention tokens:
echo %ATTN_TOKENS%
echo.
echo Expected token structure:
echo   862 variates + 4 hourly time tokens = 866
echo.
echo ==========================================================


REM ----------------------------------------------------------
REM CUDA check
REM ----------------------------------------------------------

echo Checking PyTorch CUDA availability...

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.

REM ----------------------------------------------------------
REM Check baseline checkpoint
REM ----------------------------------------------------------

if not exist "%BASELINE_CHECKPOINT%" (
    echo ==========================================================
    echo ERROR: BASELINE CHECKPOINT NOT FOUND
    echo ==========================================================
    echo %BASELINE_CHECKPOINT%
    echo.
    pause
    exit /b 1
)

echo Baseline checkpoint found.
echo.


REM ==========================================================
REM STAGE 1
REM Full iTransformer baseline test
REM ==========================================================

set "START_TIME=%TIME%"

echo ==========================================================
echo STAGE 1 - FULL iTRANSFORMER BASELINE TEST
echo Start Time: !START_TIME!
echo ==========================================================
echo.

python -u run.py ^
  --is_training 0 ^
  --model_id %BASELINE_MODEL_ID% ^
  --model iTransformer ^
  --data custom ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
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
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --num_workers 1

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo ERROR: BASELINE TEST FAILED
    echo ==========================================================
    pause
    exit /b 1
)

set "END_TIME=%TIME%"

echo.
echo ==========================================================
echo FULL BASELINE TEST COMPLETED
echo ==========================================================
echo Start: !START_TIME!
echo End:   !END_TIME!

call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo Duration: !DURATION!
echo.


REM ==========================================================
REM STAGE 2
REM Train Low-Rank R=64
REM ==========================================================

set "START_TIME=%TIME%"

echo ==========================================================
echo STAGE 2 - TRAIN GLOBAL LOW-RANK ATTENTION
echo ==========================================================
echo Rank: %RANK%
echo Attention tokens: %ATTN_TOKENS%
echo Start Time: !START_TIME!
echo ==========================================================
echo.

python -u run.py ^
  --is_training 1 ^
  --model_id %MODEL_ID% ^
  --model iTransformer ^
  --data custom ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
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
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --num_workers 1 ^
  --train_epochs 10 ^
  --patience 3 ^
  --low_rank_attention ^
  --attn_rank %RANK% ^
  --attn_tokens %ATTN_TOKENS%

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo ERROR: LOW-RANK TRAINING FAILED
    echo ==========================================================
    pause
    exit /b 1
)

set "END_TIME=%TIME%"

echo.
echo ==========================================================
echo LOW-RANK TRAINING COMPLETED
echo ==========================================================
echo Start: !START_TIME!
echo End:   !END_TIME!

call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo Duration: !DURATION!
echo.


REM ----------------------------------------------------------
REM Check newly generated checkpoint
REM ----------------------------------------------------------

if not exist "%LOWRANK_CHECKPOINT%" (
    echo ==========================================================
    echo ERROR: LOW-RANK CHECKPOINT NOT FOUND AFTER TRAINING
    echo ==========================================================
    echo Expected:
    echo %LOWRANK_CHECKPOINT%
    echo.
    pause
    exit /b 1
)

echo Low-rank checkpoint found.
echo %LOWRANK_CHECKPOINT%
echo.


REM ==========================================================
REM STAGE 3
REM Test Low-Rank R=64
REM ==========================================================

set "START_TIME=%TIME%"

echo ==========================================================
echo STAGE 3 - TEST GLOBAL LOW-RANK ATTENTION
echo ==========================================================
echo Rank: %RANK%
echo Attention tokens: %ATTN_TOKENS%
echo Start Time: !START_TIME!
echo ==========================================================
echo.

python -u run.py ^
  --is_training 0 ^
  --model_id %MODEL_ID% ^
  --model iTransformer ^
  --data custom ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
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
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --num_workers 1 ^
  --low_rank_attention ^
  --attn_rank %RANK% ^
  --attn_tokens %ATTN_TOKENS%

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo ERROR: LOW-RANK TEST FAILED
    echo ==========================================================
    pause
    exit /b 1
)

set "END_TIME=%TIME%"

echo.
echo ==========================================================
echo LOW-RANK TEST COMPLETED
echo ==========================================================
echo Start: !START_TIME!
echo End:   !END_TIME!

call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo Duration: !DURATION!
echo.


REM ==========================================================
REM FINAL SUMMARY
REM ==========================================================

echo ==========================================================
echo FINAL EXPERIMENT SUMMARY
echo ==========================================================

echo.
echo Full baseline:
echo   Model ID: %BASELINE_MODEL_ID%
echo   Checkpoint:
echo   %BASELINE_CHECKPOINT%

echo.
echo Global Low-Rank:
echo   Model ID: %MODEL_ID%
echo   Rank: %RANK%
echo   Attention tokens: %ATTN_TOKENS%
echo   Checkpoint:
echo   %LOWRANK_CHECKPOINT%

echo.
echo ==========================================================
echo IMPORTANT
echo ==========================================================
echo.
echo Compare:
echo.
echo   Full iTransformer
echo       versus
echo   Global Low-Rank R=%RANK%
echo.
echo Main metrics:
echo   MSE
echo   MAE
echo   Training duration
echo   Test/inference duration
echo   GPU memory, if measured
echo.
echo NOTE:
echo The low-rank model is trained from scratch because it
echo contains new learned token-projection parameters.
echo.
echo NOTE:
echo If PyTorch reports "Use CPU", do NOT interpret the timing
echo as GPU efficiency results.
echo.
echo ==========================================================
echo ALL STAGES COMPLETED.
echo ==========================================================

pause
exit /b 0


REM ==========================================================
REM Duration helper
REM ==========================================================

:CalculateDuration

set "START=%~1"
set "END=%~2"

for /f "tokens=1-4 delims=:.," %%a in ("%START%") do (
    set /a "start_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100"
)

for /f "tokens=1-4 delims=:.," %%a in ("%END%") do (
    set /a "end_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100"
)

set /a "elapsed_hs=end_hs-start_hs"

if !elapsed_hs! lss 0 (
    set /a "elapsed_hs+=24*60*60*100"
)

set /a "hh=elapsed_hs/360000"
set /a "rest=elapsed_hs%%360000"
set /a "mm=rest/6000"
set /a "rest=rest%%6000"
set /a "ss=rest/100"

if !hh! lss 10 set "hh=0!hh!"
if !mm! lss 10 set "mm=0!mm!"
if !ss! lss 10 set "ss=0!ss!"

set "%3=!hh!:!mm!:!ss!"

exit /b