@echo off
setlocal EnableExtensions EnableDelayedExpansion


set CUDA_VISIBLE_DEVICES=0

echo ==========================================================
echo iTransformer Traffic - Train + Rank Diagnostic
echo ==========================================================
echo Working directory:
cd
echo ==========================================================

set "START_TIME=%TIME%"

echo.
echo ==========================================
echo Experiment 1
echo Traffic 96 -> 96
echo C = 862
echo ==========================================
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
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
  --factor 1 ^
  --dropout 0.1 ^
  --embed timeF ^
  --activation gelu ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --num_workers 1 ^
  --rank_diagnostic ^
  --rank_diagnostic_split val ^
  --rank_diagnostic_batches 4 ^
  --rank_diagnostic_samples 1

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo EXPERIMENT FAILED
    echo ==========================================================
    pause
    exit /b 1
)

set "END_TIME=%TIME%"

echo.
echo ==========================================
echo Experiment completed.
echo End Time: !END_TIME!
echo ==========================================

call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo Duration: !DURATION!

echo.
echo ==========================================
echo Rank Diagnostic Result
echo ==========================================

if exist "rank_diagnostic\custom_val_DECISION.txt" (
    type "rank_diagnostic\custom_val_DECISION.txt"
) else (
    echo Diagnostic result file not found.
)

echo.
echo ==========================================
echo Output files
echo ==========================================

echo rank_diagnostic\custom_val_DECISION.txt
echo rank_diagnostic\custom_val_attention_summary.csv
echo rank_diagnostic\custom_val_representation_summary.csv
echo rank_diagnostic\custom_val_diagnostic_meta.json
echo rank_diagnostic\custom_val_singular_spectra.npz

echo.
echo ==========================================
echo ALL EXPERIMENTS COMPLETED.
echo ==========================================

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