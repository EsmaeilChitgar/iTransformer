@echo off
set CUDA_VISIBLE_DEVICES=0
setlocal enabledelayedexpansion
set model_name=iTransformer



:: ==========================================
:: Experiment 1: pred_len 24
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 1 (illness_48_24_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/illness/ ^
  --data_path national_illness.csv ^
  --model_id ili_48_24_all_ema_seed2 ^
  --model iTransformer ^
  --data custom ^
  --features M ^
  --seq_len 48 ^
  --label_len 48 ^
  --pred_len 24 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 8 ^
  --c_out 8 ^
  --des 'Exp' ^
  --num_workers 0 ^
  --itr 3

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 2: pred_len 36
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 2 (illness_48_36_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/illness/ ^
  --data_path national_illness.csv ^
  --model_id ili_48_36_all_ema_seed2 ^
  --model iTransformer ^
  --data custom ^
  --features M ^
  --seq_len 48 ^
  --label_len 48 ^
  --pred_len 36 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 8 ^
  --c_out 8 ^
  --des 'Exp' ^
  --num_workers 0 ^
  --itr 3

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 3: pred_len 36 (Duplicate)
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 3 (illness_48_36_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/illness/ ^
  --data_path national_illness.csv ^
  --model_id ili_48_36_all_ema_seed2 ^
  --model iTransformer ^
  --data custom ^
  --features M ^
  --seq_len 48 ^
  --label_len 48 ^
  --pred_len 36 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 8 ^
  --c_out 8 ^
  --des 'Exp' ^
  --num_workers 0 ^
  --itr 3

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 4: pred_len 48
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 4 (illness_48_48_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/illness/ ^
  --data_path national_illness.csv ^
  --model_id ili_48_48_all_ema_seed2 ^
  --model iTransformer ^
  --data custom ^
  --features M ^
  --seq_len 48 ^
  --label_len 48 ^
  --pred_len 48 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 8 ^
  --c_out 8 ^
  --des 'Exp' ^
  --num_workers 0 ^
  --itr 3

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

echo ==========================================
echo ALL EXPERIMENTS COMPLETED.
echo ==========================================
:: این دستور باعث می‌شود ترمینال بعد از پایان کار بسته نشود و منتظر کلید ورودی بماند
pause
goto :eof

:: ==========================================
:: Function to calculate duration (HH:MM:SS)
:: ==========================================
:CalculateDuration
set "START=%~1"
set "END=%~2"
for /f "tokens=1-4 delims=:.," %%a in ("%START%") do (set /a "start_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100")
for /f "tokens=1-4 delims=:.," %%a in ("%END%") do (set /a "end_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100")
set /a "elapsed_hs=end_hs - start_hs"
if !elapsed_hs! lss 0 set /a "elapsed_hs+=24*60*60*100"
set /a "hh=elapsed_hs / 360000", "rest=elapsed_hs %% 360000", "mm=rest / 6000", "rest=rest %% 6000", "ss=rest / 100"
if !hh! lss 10 set "hh=0!hh!"
if !mm! lss 10 set "mm=0!mm!"
if !ss! lss 10 set "ss=0!ss!"
set "%3=!hh!:!mm!:!ss!"
exit /b
