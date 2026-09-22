@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run from any directory. The repository root is one level above scripts.
pushd "%~dp0.."
if errorlevel 1 (
    echo ERROR: Cannot enter the repository root.
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "CHECKPOINT=.\checkpoints\traffic_96_96_rank_diag_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0\checkpoint.pth"
set "DATA_ROOT=.\dataset\traffic\"
set "DATA_FILE=traffic.csv"

rem Managed worktrees do not copy ignored datasets/checkpoints. Fall back to
rem the existing research checkout when the local assets are absent.
if not exist "%CHECKPOINT%" set "CHECKPOINT=C:\Users\Snapp\Documents\autoformer\paper-code\iTransformer\checkpoints\traffic_96_96_rank_diag_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0\checkpoint.pth"
if not exist "%DATA_ROOT%%DATA_FILE%" set "DATA_ROOT=C:\Users\Snapp\Documents\autoformer\paper-code\iTransformer\dataset\traffic\"

rem Stage 1 maps the broad rank curve. Stage 2 confirms the useful region on
rem more deterministic windows. Set CONFIRM_WINDOWS=0 to evaluate all windows.
set "ANALYSIS_SPLIT=val"
set "SCREEN_RANKS=8,16,32,64,128,256"
set "SCREEN_WINDOWS=32"
set "SPECTRAL_WINDOWS=32"
set "CONFIRM_RANKS=32,64,128,256"
set "CONFIRM_WINDOWS=128"
set "BOOTSTRAP_SAMPLES=20000"

set "OUTPUT_ROOT=.\rank_analysis\traffic_96_96_checkpoint"
set "SCREEN_DIR=%OUTPUT_ROOT%\screen_b%SCREEN_WINDOWS%"
set "CONFIRM_DIR=%OUTPUT_ROOT%\confirm_b%CONFIRM_WINDOWS%"
set "BENCHMARK_DIR=%OUTPUT_ROOT%\kernel_benchmarks"

echo ==========================================================
echo iTransformer Traffic 96-to-96 Low-rank Research Run
echo ==========================================================
echo Checkpoint: %CHECKPOINT%
echo Split:      %ANALYSIS_SPLIT%
echo Screen:     ranks %SCREEN_RANKS%, windows %SCREEN_WINDOWS%
echo Confirm:    ranks %CONFIRM_RANKS%, windows %CONFIRM_WINDOWS%
echo.

if not exist "%CHECKPOINT%" (
    echo ERROR: CHECKPOINT NOT FOUND
    goto :Failure
)
if not exist "%DATA_ROOT%%DATA_FILE%" (
    echo ERROR: DATASET NOT FOUND: %DATA_ROOT%%DATA_FILE%
    goto :Failure
)

call "%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available as "%PYTHON%".
    echo Activate the training environment or edit the PYTHON variable.
    goto :Failure
)

call "%PYTHON%" -c "import torch,sys; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 2 (
    echo ERROR: CUDA is unavailable. This experiment requires a CUDA PyTorch environment.
    goto :Failure
)
if errorlevel 1 goto :Failure

for %%D in ("%OUTPUT_ROOT%" "%SCREEN_DIR%" "%CONFIRM_DIR%" "%BENCHMARK_DIR%") do (
    if not exist "%%~D" mkdir "%%~D"
)

set "GIT_COMMIT=unknown"
for /f %%G in ('git rev-parse --short HEAD 2^>nul') do set "GIT_COMMIT=%%G"

(
    echo Git commit: !GIT_COMMIT!
    echo Checkpoint: %CHECKPOINT%
    echo Dataset: %DATA_ROOT%%DATA_FILE%
    echo Split: %ANALYSIS_SPLIT%
    echo Screen ranks: %SCREEN_RANKS%
    echo Screen windows: %SCREEN_WINDOWS%
    echo Confirm ranks: %CONFIRM_RANKS%
    echo Confirm windows: %CONFIRM_WINDOWS%
    echo Bootstrap samples: %BOOTSTRAP_SAMPLES%
) > "%OUTPUT_ROOT%\RUN_INFO.txt"

call "%PYTHON%" -c "import torch; print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda)); print('gpu='+torch.cuda.get_device_name(0))" > "%OUTPUT_ROOT%\PYTORCH_ENV.txt" 2>&1
nvidia-smi > "%OUTPUT_ROOT%\NVIDIA_SMI.txt" 2>&1

set "START_TIME=%TIME%"

echo.
echo ==========================================================
echo STAGE 1/3 - Broad oracle rank screen
echo Start time: !START_TIME!
echo ==========================================================

call :RunRankAnalysis "%SCREEN_RANKS%" "%SCREEN_WINDOWS%" "%SPECTRAL_WINDOWS%" "%SCREEN_DIR%" "%BOOTSTRAP_SAMPLES%"
if errorlevel 1 goto :Failure

echo.
echo ==========================================================
echo STAGE 2/3 - Paired confirmation with bootstrap intervals
echo ==========================================================

call :RunRankAnalysis "%CONFIRM_RANKS%" "%CONFIRM_WINDOWS%" "0" "%CONFIRM_DIR%" "%BOOTSTRAP_SAMPLES%"
if errorlevel 1 goto :Failure

echo.
echo ==========================================================
echo STAGE 3/3 - CUDA attention-kernel benchmarks
echo ==========================================================

for %%P in (float32 float16) do (
    for %%R in (32 64 128 256) do (
        set "BENCHMARK_FILE=%BENCHMARK_DIR%\rank_%%R_%%P.json"
        echo Benchmarking rank %%R with %%P ...
        call "%PYTHON%" tests\benchmark_variate_attention.py ^
          --device cuda ^
          --dtype %%P ^
          --tokens 866 ^
          --batch_size 16 ^
          --heads 8 ^
          --head_dim 64 ^
          --rank %%R ^
          --time_tokens 4 ^
          --warmup 20 ^
          --iterations 100 > "!BENCHMARK_FILE!" 2>&1
        if errorlevel 1 (
            type "!BENCHMARK_FILE!"
            goto :Failure
        )
        type "!BENCHMARK_FILE!"
    )
)

set "END_TIME=%TIME%"
call :CalculateDuration "!START_TIME!" "!END_TIME!" DURATION

echo.
echo ==========================================================
echo COMPLETED SUCCESSFULLY
echo End time: !END_TIME!
echo Duration: !DURATION!
echo ==========================================================
echo.
echo ---- Broad screen summary ----
type "%SCREEN_DIR%\%ANALYSIS_SPLIT%_rank_analysis_summary.json"
echo.
echo ---- Confirmation summary ----
type "%CONFIRM_DIR%\%ANALYSIS_SPLIT%_rank_analysis_summary.json"
echo.
echo Results to send for review:
echo   %OUTPUT_ROOT%\RUN_INFO.txt
echo   %SCREEN_DIR%\%ANALYSIS_SPLIT%_rank_analysis_summary.json
echo   %SCREEN_DIR%\%ANALYSIS_SPLIT%_spectral_output_errors.csv
echo   %CONFIRM_DIR%\%ANALYSIS_SPLIT%_rank_analysis_summary.json
echo   %CONFIRM_DIR%\%ANALYSIS_SPLIT%_paired_windows.csv
echo   %BENCHMARK_DIR%\rank_*_*.json
echo.
echo Please zip the complete "%OUTPUT_ROOT%" directory and send it.
echo ==========================================================

popd
pause
exit /b 0


:RunRankAnalysis
call "%PYTHON%" -u run.py ^
  --is_training 0 ^
  --model_id traffic_96_96_rank_analysis ^
  --model iTransformer ^
  --data custom ^
  --root_path "%DATA_ROOT%" ^
  --data_path "%DATA_FILE%" ^
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
  --dropout 0.1 ^
  --embed timeF ^
  --batch_size 16 ^
  --num_workers 1 ^
  --rank_analysis ^
  --checkpoint_path "%CHECKPOINT%" ^
  --rank_analysis_split "%ANALYSIS_SPLIT%" ^
  --rank_analysis_ranks "%~1" ^
  --rank_analysis_max_batches %~2 ^
  --rank_analysis_spectral_batches %~3 ^
  --rank_analysis_bootstrap_samples %~5 ^
  --rank_analysis_output_dir "%~4"
exit /b %ERRORLEVEL%


:Failure
set "FAIL_CODE=%ERRORLEVEL%"
if "%FAIL_CODE%"=="0" set "FAIL_CODE=1"
echo.
echo ==========================================================
echo ANALYSIS FAILED - exit code %FAIL_CODE%
echo ==========================================================
popd
pause
exit /b %FAIL_CODE%


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
if !elapsed_hs! lss 0 set /a "elapsed_hs+=24*60*60*100"
set /a "hh=elapsed_hs/360000"
set /a "rest=elapsed_hs%%360000"
set /a "mm=rest/6000"
set /a "rest=rest%%6000"
set /a "ss=rest/100"
if !hh! lss 10 set "hh=0!hh!"
if !mm! lss 10 set "mm=0!mm!"
if !ss! lss 10 set "ss=0!ss!"
set "%3=!hh!:!mm!:!ss!"
exit /b 0
