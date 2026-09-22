@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CUDA_VISIBLE_DEVICES=0

set "CHECKPOINT=.\checkpoints\traffic_96_96_rank_diag_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0\checkpoint.pth"

echo ==========================================================
echo iTransformer Traffic - Checkpoint Rank Analysis
echo ==========================================================
echo.
echo Checkpoint:
echo %CHECKPOINT%
echo.

if not exist "%CHECKPOINT%" (
    echo ==========================================================
    echo ERROR: CHECKPOINT NOT FOUND
    echo ==========================================================
    pause
    exit /b 1
)

echo Checkpoint found.
echo.

set "START_TIME=%TIME%"

echo ==========================================================
echo Rank Stability + Functional Rank Ablation
echo Start Time: !START_TIME!
echo ==========================================================

python -u run.py ^
  --is_training 0 ^
  --model_id traffic_96_96_rank_diag ^
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
  --train_epochs 10 ^
  --num_workers 1 ^
  --rank_analysis_only ^
  --checkpoint_path "%CHECKPOINT%" ^
  --rank_ablation ^
  --rank_ablation_batches 16 ^
  --rank_ablation_ranks "8" ^
  --rank_ablation_layerwise ^
  --rank_ablation_layer_ranks "6,11,9,6" ^
  --rank_ablation_tag "lw_6_11_9_6"

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo ANALYSIS FAILED
    echo ==========================================================
    pause
    exit /b 1
)

set "END_TIME=%TIME%"

echo.
echo ==========================================================
echo Analysis completed.
echo End Time: !END_TIME!
echo ==========================================================

call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo Duration: !DURATION!

echo.
echo ==========================================================
echo IMPORTANT RESULTS
echo ==========================================================

echo.
echo ---- Existing Rank Stability ----
if exist "rank_diagnostic\custom_val_DECISION_b32.txt" (
    type "rank_diagnostic\custom_val_DECISION_b32.txt"
) else (
    echo Stability decision file not found.
)

echo.
echo ---- Layer-wise Functional Rank Ablation ----
if exist "rank_diagnostic\custom_val_ablation_b16_lw_6_11_9_6_DECISION.txt" (
    type "rank_diagnostic\custom_val_ablation_b16_lw_6_11_9_6_DECISION.txt"
) else (
    echo Layer-wise ablation decision file not found.
)

echo.
echo ==========================================================
echo Output files
echo ==========================================================

dir /b "rank_diagnostic\custom_val_*b32*"
dir /b "rank_diagnostic\custom_val_*ablation*"

echo.
echo ==========================================================
echo ALL ANALYSIS COMPLETED.
echo ==========================================================

pause
exit /b 0


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