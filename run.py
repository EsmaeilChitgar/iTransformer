import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

# ============================================================
# PROFILING CONFIGURATION
# ============================================================

# Main switch:
# True  -> profiling enabled
# False -> profiling completely disabled
ENABLE_PROFILING = True

# ------------------------------------------------------------
# Forward-only profiling
# ------------------------------------------------------------
#
# We intentionally do NOT use torch.profiler.schedule here.
# Each selected forward pass gets its own profiler context.
# Therefore backward() and optimizer.step() are outside the trace.
#

# Number of initial forward calls that will NOT be profiled.
PROFILE_SKIP_STEPS = 3

# Additional unprofiled warm-up forward calls.
PROFILE_WARMUP_STEPS = 2

# Number of forward passes that will be individually profiled.
PROFILE_ACTIVE_STEPS = 5

# ------------------------------------------------------------
# Trace
# ------------------------------------------------------------

PROFILE_SAVE_TRACE = True

PROFILE_TRACE_DIR = './profiling'

# ------------------------------------------------------------
# Profiler detail
# ------------------------------------------------------------

PROFILE_RECORD_SHAPES = True
PROFILE_PROFILE_MEMORY = True
PROFILE_WITH_STACK = False
PROFILE_WITH_FLOPS = True

# Number of rows printed in the per-forward summary
PROFILE_TOP_OPS = 50

# ------------------------------------------------------------
# Printing
# ------------------------------------------------------------

PROFILE_PRINT_MODEL_INFO = True
PROFILE_PRINT_SHAPES = True

# ============================================================


if __name__ == '__main__':

    # ========================================================
    # Reproducibility
    # ========================================================

    fix_seed = 2023

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # ========================================================
    # Argument parser
    # ========================================================

    parser = argparse.ArgumentParser(
        description='iTransformer'
    )

    # ========================================================
    # Basic config
    # ========================================================

    parser.add_argument(
        '--is_training',
        type=int,
        required=True,
        default=1,
        help='status'
    )

    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        default='test',
        help='model id'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        default='iTransformer',
        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]'
    )

    # ========================================================
    # Data loader
    # ========================================================

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        default='custom',
        help='dataset type'
    )

    parser.add_argument(
        '--root_path',
        type=str,
        default='./dataset/electricity/',
        help='root path of the data file'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='electricity.csv',
        help='data csv file'
    )

    parser.add_argument(
        '--features',
        type=str,
        default='M',
        help='forecasting task, options:[M, S, MS]'
    )

    parser.add_argument(
        '--target',
        type=str,
        default='OT',
        help='target feature in S or MS task'
    )

    parser.add_argument(
        '--freq',
        type=str,
        default='h',
        help='freq for time features encoding'
    )

    parser.add_argument(
        '--checkpoints',
        type=str,
        default='./checkpoints/',
        help='location of the model checkpoints'
    )

    # ========================================================
    # Forecasting task
    # ========================================================

    parser.add_argument(
        '--seq_len',
        type=int,
        default=96,
        help='input sequence length'
    )

    parser.add_argument(
        '--label_len',
        type=int,
        default=48,
        help='start token length'
    )

    parser.add_argument(
        '--pred_len',
        type=int,
        default=96,
        help='prediction sequence length'
    )

    # ========================================================
    # Model define
    # ========================================================

    parser.add_argument(
        '--enc_in',
        type=int,
        default=7,
        help='encoder input size'
    )

    parser.add_argument(
        '--dec_in',
        type=int,
        default=7,
        help='decoder input size'
    )

    parser.add_argument(
        '--c_out',
        type=int,
        default=7,
        help='output size'
    )

    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='dimension of model'
    )

    parser.add_argument(
        '--n_heads',
        type=int,
        default=8,
        help='num of heads'
    )

    parser.add_argument(
        '--e_layers',
        type=int,
        default=2,
        help='num of encoder layers'
    )

    parser.add_argument(
        '--d_layers',
        type=int,
        default=1,
        help='num of decoder layers'
    )

    parser.add_argument(
        '--d_ff',
        type=int,
        default=2048,
        help='dimension of fcn'
    )

    parser.add_argument(
        '--moving_avg',
        type=int,
        default=25,
        help='window size of moving average'
    )

    parser.add_argument(
        '--factor',
        type=int,
        default=1,
        help='attn factor'
    )

    parser.add_argument(
        '--distil',
        action='store_false',
        help='whether to use distilling in encoder',
        default=True
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='dropout'
    )

    parser.add_argument(
        '--embed',
        type=str,
        default='timeF',
        help='time features encoding'
    )

    parser.add_argument(
        '--activation',
        type=str,
        default='gelu',
        help='activation'
    )

    parser.add_argument(
        '--output_attention',
        action='store_true',
        help='whether to output attention in ecoder'
    )

    parser.add_argument(
        '--do_predict',
        action='store_true',
        help='whether to predict unseen future data'
    )

    # ========================================================
    # Optimization
    # ========================================================

    parser.add_argument(
        '--num_workers',
        type=int,
        default=10,
        help='data loader num workers'
    )

    parser.add_argument(
        '--itr',
        type=int,
        default=1,
        help='experiments times'
    )

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=10,
        help='train epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size of train input data'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='early stopping patience'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='learning rate'
    )

    parser.add_argument(
        '--des',
        type=str,
        default='test',
        help='exp description'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='MSE',
        help='loss function'
    )

    parser.add_argument(
        '--lradj',
        type=str,
        default='type1',
        help='adjust learning rate'
    )

    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='use automatic mixed precision training',
        default=False
    )

    # ========================================================
    # GPU
    # ========================================================

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
        help='use gpu'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='gpu'
    )

    parser.add_argument(
        '--use_multi_gpu',
        action='store_true',
        help='use multiple gpus',
        default=False
    )

    parser.add_argument(
        '--devices',
        type=str,
        default='0,1,2,3',
        help='device ids of multile gpus'
    )

    # ========================================================
    # iTransformer
    # ========================================================

    parser.add_argument(
        '--exp_name',
        type=str,
        required=False,
        default='MTSF',
        help='experiemnt name'
    )

    parser.add_argument(
        '--channel_independence',
        type=bool,
        default=False,
        help='whether to use channel_independence mechanism'
    )

    parser.add_argument(
        '--inverse',
        action='store_true',
        help='inverse output data',
        default=False
    )

    parser.add_argument(
        '--class_strategy',
        type=str,
        default='projection',
        help='projection/average/cls_token'
    )

    parser.add_argument(
        '--target_root_path',
        type=str,
        default='./dataset/electricity/',
        help='root path of the target data'
    )

    parser.add_argument(
        '--target_data_path',
        type=str,
        default='electricity.csv',
        help='target data file'
    )

    parser.add_argument(
        '--efficient_training',
        type=bool,
        default=False,
        help='whether to use efficient_training'
    )

    parser.add_argument(
        '--use_norm',
        type=int,
        default=True,
        help='use norm and denorm'
    )

    parser.add_argument(
        '--partial_start_index',
        type=int,
        default=0,
        help='the start index of variates for partial training'
    )

    args = parser.parse_args()

    # ========================================================
    # Pass profiling configuration to model
    # ========================================================

    args.enable_profiling = ENABLE_PROFILING

    args.profile_skip_steps = (
        PROFILE_SKIP_STEPS
    )

    args.profile_warmup_steps = (
        PROFILE_WARMUP_STEPS
    )

    args.profile_active_steps = (
        PROFILE_ACTIVE_STEPS
    )

    args.profile_save_trace = (
        PROFILE_SAVE_TRACE
    )

    args.profile_trace_dir = (
        PROFILE_TRACE_DIR
    )

    args.profile_record_shapes = (
        PROFILE_RECORD_SHAPES
    )

    args.profile_memory = (
        PROFILE_PROFILE_MEMORY
    )

    args.profile_with_stack = (
        PROFILE_WITH_STACK
    )

    args.profile_with_flops = (
        PROFILE_WITH_FLOPS
    )

    args.profile_top_ops = (
        PROFILE_TOP_OPS
    )

    args.profile_print_model_info = (
        PROFILE_PRINT_MODEL_INFO
    )

    args.profile_print_shapes = (
        PROFILE_PRINT_SHAPES
    )

    # ========================================================
    # GPU setup
    # ========================================================

    args.use_gpu = (
        True
        if torch.cuda.is_available()
        and args.use_gpu
        else False
    )

    if args.use_gpu and args.use_multi_gpu:

        args.devices = args.devices.replace(
            ' ',
            ''
        )

        device_ids = args.devices.split(',')

        args.device_ids = [
            int(id_)
            for id_ in device_ids
        ]

        args.gpu = args.device_ids[0]

    # ========================================================
    # Profiling diagnostics
    # ========================================================

    print('\n' + '=' * 100)
    print('PROFILER / CUDA DIAGNOSTICS')
    print('=' * 100)

    print(
        'PyTorch version       :',
        torch.__version__
    )

    print(
        'Torch CUDA version    :',
        torch.version.cuda
    )

    print(
        'CUDA available        :',
        torch.cuda.is_available()
    )

    try:

        print(
            'Kineto available      :',
            torch.profiler.kineto_available()
        )

    except Exception as e:

        print(
            'Kineto available      : ERROR'
        )

        print(
            '  ',
            repr(e)
        )

    try:

        print(
            'Supported activities  :',
            torch.profiler.supported_activities()
        )

    except Exception as e:

        print(
            'Supported activities  : ERROR'
        )

        print(
            '  ',
            repr(e)
        )

    if torch.cuda.is_available():

        try:

            print(
                'CUDA device           :',
                torch.cuda.get_device_name(
                    0
                )
            )

            print(
                'CUDA device count     :',
                torch.cuda.device_count()
            )

            print(
                'Current CUDA device   :',
                torch.cuda.current_device()
            )

        except Exception as e:

            print(
                'CUDA device info      : ERROR'
            )

            print(
                '  ',
                repr(e)
            )

    print('=' * 100)

    # ========================================================
    # Profiling configuration
    # ========================================================

    print('\n' + '=' * 100)
    print('iTRANSFORMER PROFILING CONFIGURATION')
    print('=' * 100)

    print(
        f'ENABLE_PROFILING      : '
        f'{args.enable_profiling}'
    )

    if args.enable_profiling:

        print(
            f'PROFILE_SKIP_STEPS    : '
            f'{args.profile_skip_steps}'
        )

        print(
            f'PROFILE_WARMUP_STEPS  : '
            f'{args.profile_warmup_steps}'
        )

        print(
            f'PROFILE_ACTIVE_STEPS  : '
            f'{args.profile_active_steps}'
        )

        print(
            f'PROFILE_RECORD_SHAPES : '
            f'{args.profile_record_shapes}'
        )

        print(
            f'PROFILE_MEMORY        : '
            f'{args.profile_memory}'
        )

        print(
            f'PROFILE_WITH_FLOPS    : '
            f'{args.profile_with_flops}'
        )

        print(
            f'PROFILE_TRACE_DIR     : '
            f'{args.profile_trace_dir}'
        )

    print('=' * 100 + '\n')

    # ========================================================
    # Experiment args
    # ========================================================

    print('Args in experiment:')
    print(args)

    # ========================================================
    # Select experiment
    # ========================================================

    if args.exp_name == 'partial_train':

        Exp = Exp_Long_Term_Forecast_Partial

    else:

        Exp = Exp_Long_Term_Forecast

    # ========================================================
    # Training
    # ========================================================

    if args.is_training:

        for ii in range(args.itr):

            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy,
                ii
            )

            exp = Exp(args)

            print(
                '>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
                    setting
                )
            )

            exp.train(setting)

            print(
                '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                    setting
                )
            )

            exp.test(setting)

            if args.do_predict:

                print(
                    '>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                        setting
                    )
                )

                exp.predict(
                    setting,
                    True
                )

            torch.cuda.empty_cache()

    # ========================================================
    # Testing
    # ========================================================

    else:

        ii = 0

        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            ii
        )

        exp = Exp(args)

        print(
            '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                setting
            )
        )

        exp.test(
            setting,
            test=1
        )

        torch.cuda.empty_cache()