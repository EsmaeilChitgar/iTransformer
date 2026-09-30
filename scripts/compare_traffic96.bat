@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0.."
if errorlevel 1 exit /b 1

REM Article-matched first comparison screen: Traffic 96->96, 862 variables.
REM Dense and ILRA are ALREADY available in the paper. Run ISAB and Luna first.
REM Official mechanisms treat all tokens uniformly; bypass is a separately
REM labeled forecasting adaptation, intentionally NOT mixed in the initial test.
set "PYTHON=python"
set "DATA_ROOT=.\dataset\traffic\"
set "DATA_FILE=traffic.csv"
set "OUT_DIR=.\comparison_logs"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
REM Hard fail if someone labels a run seed2023 without actually fixing the RNG seed.
%PYTHON% -c "import pathlib,re,sys; t=pathlib.Path('run.py').read_text(); m=re.search(r'(?m)^\s*fix_seed\s*=\s*(\d+)\s*$',t); print('actual seed=',m.group(1) if m else 'unresolved'); sys.exit(0 if m and m.group(1)=='2023' else 2)"
if errorlevel 1 (
  echo ERROR: run.py does not fix seed=2023. Match all baselines first.
  popd
  exit /b 1
)

if not exist "%DATA_ROOT%%DATA_FILE%" (
  echo ERROR: traffic.csv missing in %DATA_ROOT%
  popd
  exit /b 1
)

%PYTHON% tests\audit_data_factory.py --strict > "%OUT_DIR%\data_factory_audit.log" 2>&1
if errorlevel 1 (
  type "%OUT_DIR%\data_factory_audit.log"
  popd
  exit /b 1
)

%PYTHON% tests\verify_latest_ilra_source.py > "%OUT_DIR%\ilra_source_verification.log" 2>&1
if errorlevel 1 (
  type "%OUT_DIR%\ilra_source_verification.log"
  popd
  exit /b 1
)

%PYTHON% tests\test_reference_parity.py > "%OUT_DIR%\reference_equations.log" 2>&1
if errorlevel 1 (
  type "%OUT_DIR%\reference_equations.log"
  popd
  exit /b 1
)

%PYTHON% scripts\comparison_preflight.py > "%OUT_DIR%\preflight.log" 2>&1
if errorlevel 1 (
  type "%OUT_DIR%\preflight.log"
  popd
  exit /b 1
)

for %%M in (isab luna) do (
  echo ------------------------------------------------------------
  echo Training %%M Traffic 96->96 rank 8 seed 2023
  echo ------------------------------------------------------------
  %PYTHON% -u run.py ^
    --is_training 1 ^
    --model_id traffic_96_96_comp_%%M_r8_seed2023 ^
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
    --factor 1 ^
    --dropout 0.1 ^
    --activation gelu ^
    --embed timeF ^
    --use_norm 1 ^
    --batch_size 16 ^
    --learning_rate 0.001 ^
    --train_epochs 10 ^
    --patience 3 ^
    --num_workers 1 ^
    --itr 1 ^
    --des compare ^
    --attn_rank 8 ^
    --compare_attention %%M ^
    --isab_layernorm 0 ^
    --comparison_time_tokens 4 > "%OUT_DIR%\traffic96_%%M_r8.log" 2>&1
  if errorlevel 1 (
    echo ERROR: %%M. Log:
    type "%OUT_DIR%\traffic96_%%M_r8.log"
    popd
    exit /b 1
  )
  findstr /i /c:"mse:" "%OUT_DIR%\traffic96_%%M_r8.log"
)

echo ------------------------------------------------------------
echo COMPLETED. Send complete comparison_logs and result_long_term_forecast.txt.
echo ------------------------------------------------------------
popd
exit /b 0
