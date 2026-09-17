@echo off
set CUDA_VISIBLE_DEVICES=0
setlocal enabledelayedexpansion
set "model_name=iTransformer"

:: ==========================================
:: Experiment 1: weather_96_96
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 1 (weather_96_96_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/weather/ ^
  --data_path weather_all_ema.csv ^
  --model_id weather_96_96_all_ema_seed2 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 22 ^
  --dec_in 22 ^
  --c_out 22 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --num_workers 2 ^
  --itr 1

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 2: weather_96_192
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 2 (weather_96_192_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/weather/ ^
  --data_path weather_all_ema.csv ^
  --model_id weather_96_192_all_ema_seed2 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 192 ^
  --e_layers 3 ^
  --enc_in 22 ^
  --dec_in 22 ^
  --c_out 22 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --num_workers 2 ^
  --itr 1

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 3: weather_96_336
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 3 (weather_96_336_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/weather/ ^
  --data_path weather_all_ema.csv ^
  --model_id weather_96_336_all_ema_seed2 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 336 ^
  --e_layers 3 ^
  --enc_in 22 ^
  --dec_in 22 ^
  --c_out 22 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --num_workers 2 ^
  --itr 1

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.

:: ==========================================
:: Experiment 4: weather_96_720
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 4 (weather_96_720_all_ema_seed2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/weather/ ^
  --data_path weather_all_ema.csv ^
  --model_id weather_96_720_all_ema_seed2 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 720 ^
  --e_layers 3 ^
  --enc_in 22 ^
  --dec_in 22 ^
  --c_out 22 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --num_workers 2 ^
  --itr 1

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.


:: ==========================================
:: Experiment 1: weather_96_96
:: ==========================================
set "START_TIME=%TIME%"
echo ==========================================
echo Running Experiment 1 (weather_96_96_all_ema_seed2_v2)...
echo Start Time: !START_TIME!
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/weather/ ^
  --data_path weather_all_ema.csv ^
  --model_id weather_96_96_all_ema_seed2_v2 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 22 ^
  --dec_in 22 ^
  --c_out 22 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --num_workers 2 ^
  --itr 1

set "END_TIME=%TIME%"
echo End Time: !END_TIME!
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION
echo Duration: !DURATION!
echo.


echo ==========================================
echo ALL EXPERIMENTS COMPLETED.
echo ==========================================
pause
goto :eof

:: ==========================================
:: Function: Calculate duration (HH:MM:SS)
:: ==========================================
:CalculateDuration
set "START=%~1"
set "END=%~2"

for /f "tokens=1-4 delims=:.," %%a in ("%START%") do (
  set /a "start_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100"
)
for /f "tokens=1-4 delims=:.," %%a in ("%END%") do (
  set /a "end_hs=(((1%%a*60)+1%%b)*60+1%%c)*100+1%%d-36610100"
)

set /a "elapsed_hs=end_hs - start_hs"
if !elapsed_hs! lss 0 set /a "elapsed_hs+=24*60*60*100"

set /a "hh=elapsed_hs / 360000"
set /a "rest=elapsed_hs %% 360000"
set /a "mm=rest / 6000"
set /a "rest=rest %% 6000"
set /a "ss=rest / 100"

if !hh! lss 10 set "hh=0!hh!"
if !mm! lss 10 set "mm=0!mm!"
if !ss! lss 10 set "ss=0!ss!"

set "%3=!hh!:!mm!:!ss!"
exit /b
