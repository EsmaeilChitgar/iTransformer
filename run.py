import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of the data file')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the target data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    # Rank diagnostic
    parser.add_argument('--rank_diagnostic', action='store_true',
                        help='run rank/spectral diagnostics on the trained best checkpoint')
    parser.add_argument('--rank_diagnostic_split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='split used for rank diagnostic')
    parser.add_argument('--rank_diagnostic_batches', type=int, default=4,
                        help='number of batches used for rank diagnostic')
    parser.add_argument('--rank_diagnostic_samples', type=int, default=1,
                        help='number of samples per batch analyzed for rank diagnostic')
    parser.add_argument('--rank_analysis_only', action='store_true',
                        help='run rank analysis using an existing checkpoint without training')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='checkpoint path used for rank analysis')
    parser.add_argument('--rank_ablation', action='store_true',
                        help='run functional rank ablation')
    parser.add_argument('--rank_ablation_batches', type=int, default=4,
                        help='number of validation batches used for rank ablation')
    parser.add_argument('--rank_ablation_ranks', type=str, default='1,2,4,8,16,32',
                        help='comma-separated attention ranks')

    parser.add_argument('--rank_ablation_layerwise', action='store_true',
                        help='use layer-wise attention ranks during rank ablation')
    parser.add_argument('--rank_ablation_layer_ranks', type=str, default='6,11,9,6',
                        help='comma-separated rank for each encoder layer')
    parser.add_argument('--rank_ablation_tag', type=str, default='',
                        help='suffix for rank ablation result files')
    parser.add_argument(
        '--rank_ablation_paired',
        action='store_true',
        help='compare Full, Global-Rank, and Layer-wise rank on the same windows'
    )
    parser.add_argument(
        '--rank_ablation_headwise',
        action='store_true',
        help='paired Full vs Global-Rank vs Head-wise rank ablation'
    )

    parser.add_argument(
        '--rank_ablation_head_csv',
        type=str,
        default='./rank_diagnostic/custom_val_attention_summary_b32.csv',
        help='CSV containing per-layer/head r95 values'
    )

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.rank_analysis_only:
        if not args.checkpoint_path:
            raise ValueError('checkpoint_path is required when rank_analysis_only is enabled')

        exp = Exp(args)

        print('>>>>>>> loading checkpoint : {} <<<<<<<<<<<<<<<<<<<<<<'.format(
            args.checkpoint_path
        ))

        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=exp.device
        )

        exp.model.load_state_dict(checkpoint)

        print('>>>>>>> checkpoint loaded <<<<<<<<<<<<<<<<<<<<<<')

        if args.rank_diagnostic:
            print('>>>>>>> start rank diagnostic : {} <<<<<<<<<<<<<<<<<<<<<<'.format(
                args.rank_diagnostic_split
            ))

            exp.rank_diagnostic(
                flag=args.rank_diagnostic_split,
                max_batches=args.rank_diagnostic_batches,
                max_samples=args.rank_diagnostic_samples
            )

            print('>>>>>>> rank diagnostic finished <<<<<<<<<<<<<<<<<<<<<<')

        if args.rank_ablation_paired:
            exp.rank_ablation_paired(
                flag='val',
                max_batches=args.rank_ablation_batches,
                global_rank=8,
                layer_ranks=args.rank_ablation_layer_ranks
            )

        if args.rank_ablation_headwise:
            print(
                '>>>>>>> start paired head-wise rank ablation '
                '<<<<<<<<<<<<<<<<<<<<<<'
            )

            exp.rank_ablation_paired_headwise(
                flag='val',
                max_batches=args.rank_ablation_batches,
                global_rank=8,
                attention_summary_csv=args.rank_ablation_head_csv
            )

            print(
                '>>>>>>> paired head-wise rank ablation finished '
                '<<<<<<<<<<<<<<<<<<<<'
            )

        if args.rank_ablation:
            print('>>>>>>> start rank ablation <<<<<<<<<<<<<<<<<<<<<<')

            exp.rank_ablation(
                flag=args.rank_diagnostic_split,
                max_batches=args.rank_ablation_batches,
                ranks=args.rank_ablation_ranks
            )

            print('>>>>>>> rank ablation finished <<<<<<<<<<<<<<<<<<<<<<')

        torch.cuda.empty_cache()

    elif args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
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
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.rank_diagnostic:
                print('>>>>>>>start rank diagnostic : {}>>>>>>>>>>>>>>>>>>>>'.format(
                    args.rank_diagnostic_split
                ))
                exp.rank_diagnostic(
                    flag=args.rank_diagnostic_split,
                    max_batches=args.rank_diagnostic_batches,
                    max_samples=args.rank_diagnostic_samples
                )
                print('>>>>>>>rank diagnostic finished <<<<<<<<<<<<<<<<<<<<<<')

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
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
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
