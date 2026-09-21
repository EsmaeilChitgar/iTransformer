from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
import json
import math
from collections import defaultdict

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def rank_diagnostic(
            self,
            flag='val',
            max_batches=4,
            max_samples=1
    ):
        """
        Analyze the learned representation and attention spectra of the
        trained iTransformer checkpoint.

        This is a diagnostic experiment only.
        It does NOT modify model weights.
        """

        print('\n' + '=' * 90)
        print('RANK / SPECTRAL DIAGNOSTIC')
        print('=' * 90)
        print('Split:', flag)
        print('Max batches:', max_batches)
        print('Max samples per batch:', max_samples)

        data_set, data_loader = self._get_data(flag=flag)

        model = self.model.module if hasattr(
            self.model, 'module'
        ) else self.model

        model.eval()
        model.set_diagnostic_mode(True)

        # -------------------------------------------------------------
        # Helpers
        # -------------------------------------------------------------

        def rank_from_energy(cumulative_energy, threshold):
            idx = torch.nonzero(
                cumulative_energy >= threshold,
                as_tuple=False
            )

            if idx.numel() == 0:
                return int(cumulative_energy.numel())

            return int(idx[0].item() + 1)

        def matrix_stats(matrix, topk_values=(8, 32, 64, 128)):
            """
            matrix: [M, N]

            Returns singular-spectrum and concentration statistics.
            """

            matrix = matrix.float()

            singular_values = torch.linalg.svdvals(
                matrix
            )

            energy = singular_values.square()

            total_energy = energy.sum().clamp_min(1e-12)

            normalized_energy = (
                energy / total_energy
            )

            cumulative_energy = torch.cumsum(
                normalized_energy,
                dim=0
            )

            r90 = rank_from_energy(
                cumulative_energy,
                0.90
            )

            r95 = rank_from_energy(
                cumulative_energy,
                0.95
            )

            r99 = rank_from_energy(
                cumulative_energy,
                0.99
            )

            # Stable rank:
            # ||A||_F^2 / ||A||_2^2
            stable_rank = (
                energy.sum()
                / singular_values[0].square().clamp_min(1e-12)
            )

            # Entropy-based effective rank.
            p = singular_values / (
                singular_values.sum().clamp_min(1e-12)
            )

            entropy = -(
                p.clamp_min(1e-12)
                * p.clamp_min(1e-12).log()
            ).sum()

            effective_rank = torch.exp(entropy)

            result = {
                'singular_values': singular_values.numpy(),
                'r90': r90,
                'r95': r95,
                'r99': r99,
                'stable_rank': float(stable_rank.item()),
                'effective_rank': float(
                    effective_rank.item()
                )
            }

            for k in topk_values:

                if k <= matrix.shape[-1]:
                    result[
                        f'energy_r{k}'
                    ] = float(
                        cumulative_energy[k - 1].item()
                    )

            return result

        def attention_stats(A):
            """
            A: [N, N]

            Attention is row-normalized.
            """

            stats = matrix_stats(A)

            N = A.shape[-1]

            A_safe = A.clamp_min(1e-12)

            row_entropy = -(
                A_safe * A_safe.log()
            ).sum(dim=-1).mean()

            normalized_entropy = (
                row_entropy / math.log(N)
            )

            stats['normalized_attention_entropy'] = float(
                normalized_entropy.item()
            )

            for k in [8, 32, 64, 128]:

                if k <= N:

                    top_mass = torch.topk(
                        A,
                        k=k,
                        dim=-1
                    ).values.sum(dim=-1).mean()

                    stats[
                        f'top{k}_mass'
                    ] = float(top_mass.item())

            return stats

        # -------------------------------------------------------------
        # Storage
        # -------------------------------------------------------------

        representation_rows = []
        attention_rows = []

        spectra = {}

        batches_processed = 0

        # -------------------------------------------------------------
        # Diagnostic forward passes
        # -------------------------------------------------------------

        try:

            with torch.no_grad():

                for batch_idx, (
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark
                ) in enumerate(data_loader):

                    if batch_idx >= max_batches:
                        break

                    batch_x = batch_x.float().to(
                        self.device
                    )

                    batch_y = batch_y.float().to(
                        self.device
                    )

                    if (
                            'PEMS' in self.args.data
                            or 'Solar' in self.args.data
                    ):
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = (
                            batch_x_mark
                            .float()
                            .to(self.device)
                        )

                        batch_y_mark = (
                            batch_y_mark
                            .float()
                            .to(self.device)
                        )

                    dec_inp = torch.zeros_like(
                        batch_y[
                            :,
                            -self.args.pred_len:,
                            :
                        ]
                    ).float()

                    dec_inp = torch.cat(
                        [
                            batch_y[
                                :,
                                :self.args.label_len,
                                :
                            ],
                            dec_inp
                        ],
                        dim=1
                    ).float().to(self.device)

                    # Forward pass.
                    self.model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark
                    )

                    layer_outputs = model.last_layer_outputs
                    attentions = model.last_attentions

                    current_samples = min(
                        max_samples,
                        batch_x.shape[0]
                    )

                    for layer_idx in range(
                        len(layer_outputs)
                    ):

                        E_batch = layer_outputs[
                            layer_idx
                        ][:current_samples]

                        A_batch = attentions[
                            layer_idx
                        ]

                        # -------------------------------------------------
                        # Representation analysis
                        # -------------------------------------------------

                        for sample_idx in range(
                            current_samples
                        ):

                            E = E_batch[
                                sample_idx
                            ]

                            E_stats = matrix_stats(
                                E
                            )

                            E_stats[
                                'layer'
                            ] = layer_idx + 1

                            E_stats[
                                'sample'
                            ] = sample_idx

                            E_stats[
                                'batch'
                            ] = batch_idx

                            representation_rows.append(
                                E_stats
                            )

                            spectra[
                                f'E_L{layer_idx + 1}_B{batch_idx}_S{sample_idx}'
                            ] = E_stats[
                                'singular_values'
                            ]

                        # -------------------------------------------------
                        # Attention analysis
                        # -------------------------------------------------

                        if A_batch is None:
                            continue

                        A_batch = A_batch[
                            :current_samples
                        ]

                        # A_batch:
                        # [samples, heads, N, N]

                        for sample_idx in range(
                            current_samples
                        ):

                            A_sample = A_batch[
                                sample_idx
                            ]

                            for head_idx in range(
                                A_sample.shape[0]
                            ):

                                A = A_sample[
                                    head_idx
                                ]

                                A_stats = attention_stats(
                                    A
                                )

                                A_stats[
                                    'layer'
                                ] = layer_idx + 1

                                A_stats[
                                    'head'
                                ] = head_idx + 1

                                A_stats[
                                    'sample'
                                ] = sample_idx

                                A_stats[
                                    'batch'
                                ] = batch_idx

                                attention_rows.append(
                                    A_stats
                                )

                                spectra[
                                    f'A_L{layer_idx + 1}_H{head_idx + 1}_B{batch_idx}_S{sample_idx}'
                                ] = A_stats[
                                    'singular_values'
                                ]

                    batches_processed += 1

                    print(
                        f'Diagnostic batch '
                        f'{batches_processed}/{max_batches} completed.'
                    )

        finally:
            model.set_diagnostic_mode(False)
            model.eval()

        # -------------------------------------------------------------
        # Aggregate summaries
        # -------------------------------------------------------------

        def aggregate(rows, include_head=False):

            groups = defaultdict(list)

            for row in rows:

                if include_head:
                    key = (
                        row['layer'],
                        row['head']
                    )
                else:
                    key = (
                        row['layer'],
                    )

                groups[key].append(row)

            summary = []

            for key, group in sorted(
                    groups.items()
            ):

                row = {
                    'layer': key[0],
                    'num_observations': len(group)
                }

                if include_head:
                    row['head'] = key[1]

                numeric_keys = [
                    k
                    for k in group[0].keys()
                    if k not in {
                        'singular_values',
                        'layer',
                        'head',
                        'sample',
                        'batch'
                    }
                ]

                for metric_name in numeric_keys:

                    values = np.array(
                        [
                            item[metric_name]
                            for item in group
                        ],
                        dtype=np.float64
                    )

                    row[metric_name] = float(
                        np.mean(values)
                    )

                    row[
                        metric_name + '_std'
                    ] = float(
                        np.std(values)
                    )

                    row[
                        metric_name + '_min'
                    ] = float(
                        np.min(values)
                    )

                    row[
                        metric_name + '_max'
                    ] = float(
                        np.max(values)
                    )

                summary.append(row)

            return summary

        representation_summary = aggregate(
            representation_rows,
            include_head=False
        )

        attention_summary = aggregate(
            attention_rows,
            include_head=True
        )

        # -------------------------------------------------------------
        # Decision screen
        # -------------------------------------------------------------

        attention_r95_ratios = []

        for row in attention_summary:
            attention_r95_ratios.append(
                row['r95'] / float(
                    self.args.enc_in
                )
            )

        if attention_r95_ratios:

            median_r95_ratio = float(
                np.median(
                    attention_r95_ratios
                )
            )

            min_r95_ratio = float(
                np.min(
                    attention_r95_ratios
                )
            )

            max_r95_ratio = float(np.max(attention_r95_ratios))

            if median_r95_ratio <= 0.20:
                rank_screen = 'LOW_RANK_SCREEN'

            elif median_r95_ratio >= 0.80:
                rank_screen = 'HIGH_RANK_SCREEN'

            else:
                rank_screen = 'INTERMEDIATE_RANK_SCREEN'

            if (
                    min_r95_ratio > 0
                    and max_r95_ratio / min_r95_ratio >= 2.0
            ):
                heterogeneity_screen = 'HIGH_HEAD_LAYER_HETEROGENEITY'

            else:
                heterogeneity_screen = 'NO_STRONG_HEAD_LAYER_HETEROGENEITY'
        else:
            median_r95_ratio = None
            min_r95_ratio = None
            max_r95_ratio = None
            rank_screen = 'NO_ATTENTION_DATA'
            heterogeneity_screen = 'NO_DATA'

        # -------------------------------------------------------------
        # Save outputs
        # -------------------------------------------------------------

        save_dir = './rank_diagnostic'

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        # Representation CSV
        representation_csv = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}_representation_summary_b{max_batches}.csv'
        )

        if representation_summary:

            fieldnames = list(
                representation_summary[0].keys()
            )

            with open(
                    representation_csv,
                    'w',
                    newline=''
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames
                )

                writer.writeheader()
                writer.writerows(
                    representation_summary
                )

        # Attention CSV
        attention_csv = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}_attention_summary_b{max_batches}.csv'
        )

        if attention_summary:

            fieldnames = list(
                attention_summary[0].keys()
            )

            with open(
                    attention_csv,
                    'w',
                    newline=''
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames
                )

                writer.writeheader()
                writer.writerows(
                    attention_summary
                )

        # Raw spectra
        spectrum_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}_singular_spectra_b{max_batches}.npz'
        )

        np.savez_compressed(
            spectrum_path,
            **spectra
        )

        # Meta JSON
        meta = {
            'dataset': self.args.data,
            'split': flag,
            'enc_in': self.args.enc_in,
            'd_model': self.args.d_model,
            'n_heads': self.args.n_heads,
            'e_layers': self.args.e_layers,
            'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len,
            'batches_processed': batches_processed,
            'samples_per_batch': max_samples,
            'median_attention_r95_ratio': median_r95_ratio,
            'min_attention_r95_ratio': min_r95_ratio,
            'max_attention_r95_ratio': max_r95_ratio,
            'rank_screen': rank_screen,
            'heterogeneity_screen': heterogeneity_screen
        }

        meta_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}_diagnostic_meta_b{max_batches}.json'
        )

        with open(
                meta_path,
                'w'
        ) as f:

            json.dump(
                meta,
                f,
                indent=2
            )

        # Decision text
        decision_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}_DECISION_b{max_batches}.txt'
        )

        with open(
                decision_path,
                'w'
        ) as f:

            f.write(
                'RANK DIAGNOSTIC SCREEN\n'
            )
            f.write(
                '======================\n\n'
            )
            f.write(
                f'Rank screen: {rank_screen}\n'
            )
            f.write(
                f'Heterogeneity screen: '
                f'{heterogeneity_screen}\n'
            )
            f.write(
                f'Median attention r95/N: '
                f'{median_r95_ratio}\n'
            )
            f.write(
                f'Min attention r95/N: '
                f'{min_r95_ratio}\n'
            )
            f.write(
                f'Max attention r95/N: '
                f'{max_r95_ratio}\n'
            )
            f.write(
                '\nThese are diagnostic screening rules, '
                'not statistical significance claims.\n'
            )

        # -------------------------------------------------------------
        # Console summary
        # -------------------------------------------------------------

        print('\n' + '=' * 90)
        print('RANK DIAGNOSTIC SUMMARY')
        print('=' * 90)

        print(
            f'Median attention r95/N: '
            f'{median_r95_ratio}'
        )

        print(
            f'Min attention r95/N: '
            f'{min_r95_ratio}'
        )

        print(
            f'Max attention r95/N: '
            f'{max_r95_ratio}'
        )

        print(
            f'Rank screen: {rank_screen}'
        )

        print(
            f'Heterogeneity screen: '
            f'{heterogeneity_screen}'
        )

        print('\nSaved files:')

        print(representation_csv)
        print(attention_csv)
        print(spectrum_path)
        print(meta_path)
        print(decision_path)

        print('=' * 90)

    def rank_ablation(
            self,
            flag='val',
            max_batches=4,
            ranks='1,2,4,8,16,32'
    ):
        """
        Functional rank ablation using an already-trained checkpoint.

        rank=0:
            original full attention.

        rank>0:
            replace each attention matrix with its truncated
            rank-r SVD approximation.

        This experiment measures forecasting sensitivity to
        attention rank. It is NOT a runtime benchmark because
        the full attention matrix and its SVD are still computed.
        """

        model = self.model.module if hasattr(
            self.model,
            'module'
        ) else self.model

        rank_values = [0]

        if isinstance(ranks, str):
            for value in ranks.split(','):
                value = value.strip()

                if value:
                    rank_values.append(
                        int(value)
                    )
        else:
            rank_values.extend(ranks)

        rank_values = sorted(
            list(set(rank_values))
        )

        print('\n' + '=' * 90)
        print('FUNCTIONAL RANK ABLATION')
        print('=' * 90)
        print('Split:', flag)
        print('Batches:', max_batches)
        print('Ranks:', rank_values)
        print('Samples per batch: 1')
        print('=' * 90)

        data_set, data_loader = self._get_data(
            flag=flag
        )

        model.eval()

        mse_values = {
            rank: []
            for rank in rank_values
        }

        mae_values = {
            rank: []
            for rank in rank_values
        }

        try:

            with torch.no_grad():

                for batch_idx, (
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark
                ) in enumerate(data_loader):

                    if batch_idx >= max_batches:
                        break

                    # -------------------------------------------------
                    # Use exactly ONE sample from the DataLoader batch.
                    # This keeps this screening experiment lightweight.
                    # -------------------------------------------------

                    batch_x = batch_x[:1].float().to(
                        self.device
                    )

                    batch_y = batch_y[:1].float().to(
                        self.device
                    )

                    if (
                            'PEMS' in self.args.data
                            or 'Solar' in self.args.data
                    ):
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = (
                            batch_x_mark[:1]
                            .float()
                            .to(self.device)
                        )

                        batch_y_mark = (
                            batch_y_mark[:1]
                            .float()
                            .to(self.device)
                        )

                    # -------------------------------------------------
                    # Decoder input
                    # -------------------------------------------------

                    dec_inp = torch.zeros_like(
                        batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                        ]
                    ).float()

                    dec_inp = torch.cat(
                        [
                            batch_y[
                            :,
                            :self.args.label_len,
                            :
                            ],
                            dec_inp
                        ],
                        dim=1
                    ).float().to(self.device)

                    # -------------------------------------------------
                    # Evaluate every requested rank
                    # -------------------------------------------------

                    for rank in rank_values:

                        model.set_rank_ablation(
                            rank
                        )

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                        # In the current configuration output_attention
                        # is False, but keep this for compatibility.
                        if self.args.output_attention:
                            outputs = outputs[0]

                        f_dim = (
                            -1
                            if self.args.features == 'MS'
                            else 0
                        )

                        outputs = outputs[
                                  :,
                                  -self.args.pred_len:,
                                  f_dim:
                                  ]

                        true = batch_y[
                               :,
                               -self.args.pred_len:,
                               f_dim:
                               ]

                        # -------------------------------------------------
                        # Forecasting metrics
                        # -------------------------------------------------

                        batch_mse = torch.mean(
                            (outputs - true) ** 2
                        )

                        batch_mae = torch.mean(
                            torch.abs(outputs - true)
                        )

                        mse_values[rank].append(
                            float(
                                batch_mse.item()
                            )
                        )

                        mae_values[rank].append(
                            float(
                                batch_mae.item()
                            )
                        )

                    # Always restore full attention after each batch.
                    model.set_rank_ablation(0)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(
                        'Ablation batch '
                        f'{batch_idx + 1}/{max_batches} completed.'
                    )

        finally:

            # Always leave the model in normal full-attention mode.
            model.set_rank_ablation(0)
            model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Aggregate results
        # -------------------------------------------------------------

        rows = []

        full_mse = float(
            np.mean(
                mse_values[0]
            )
        )

        full_mae = float(
            np.mean(
                mae_values[0]
            )
        )

        print('\n' + '=' * 90)
        print('FUNCTIONAL RANK ABLATION RESULTS')
        print('=' * 90)

        for rank in rank_values:
            mse = float(
                np.mean(
                    mse_values[rank]
                )
            )

            mae = float(
                np.mean(
                    mae_values[rank]
                )
            )

            mse_gap = (
                    100.0
                    * (mse - full_mse)
                    / max(
                abs(full_mse),
                1e-12
            )
            )

            mae_gap = (
                    100.0
                    * (mae - full_mae)
                    / max(
                abs(full_mae),
                1e-12
            )
            )

            row = {
                'rank': rank,
                'mse': mse,
                'mae': mae,
                'mse_gap_percent_vs_full': mse_gap,
                'mae_gap_percent_vs_full': mae_gap,
                'num_batches': len(
                    mse_values[rank]
                )
            }

            rows.append(row)

            print(
                f'Rank {rank:>3}: '
                f'MSE={mse:.8f}, '
                f'MAE={mae:.8f}, '
                f'MSE gap={mse_gap:+.3f}%, '
                f'MAE gap={mae_gap:+.3f}%'
            )

        # -------------------------------------------------------------
        # Functional screening
        # -------------------------------------------------------------

        non_full_rows = [
            row
            for row in rows
            if row['rank'] != 0
        ]

        within_1_percent = [
            row['rank']
            for row in non_full_rows
            if row['mse_gap_percent_vs_full'] <= 1.0
               and row['mse_gap_percent_vs_full'] >= -1.0
        ]

        within_3_percent = [
            row['rank']
            for row in non_full_rows
            if row['mse_gap_percent_vs_full'] <= 3.0
               and row['mse_gap_percent_vs_full'] >= -3.0
        ]

        if within_1_percent:

            screen = (
                'STRONG_FUNCTIONAL_LOW_RANK_EVIDENCE'
            )

        elif within_3_percent:

            screen = (
                'PROMISING_BUT_REQUIRES_FURTHER_VALIDATION'
            )

        else:

            screen = (
                'WEAK_FUNCTIONAL_LOW_RANK_EVIDENCE'
            )

        # Smallest rank within 1% and 3%.
        smallest_rank_1 = (
            min(within_1_percent)
            if within_1_percent
            else None
        )

        smallest_rank_3 = (
            min(within_3_percent)
            if within_3_percent
            else None
        )

        # -------------------------------------------------------------
        # Save CSV
        # -------------------------------------------------------------

        save_dir = './rank_diagnostic'

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        suffix = (
            f'_ablation_b{max_batches}'
        )

        csv_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}{suffix}.csv'
        )

        fieldnames = [
            'rank',
            'mse',
            'mae',
            'mse_gap_percent_vs_full',
            'mae_gap_percent_vs_full',
            'num_batches'
        ]

        with open(
                csv_path,
                'w',
                newline=''
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

        # -------------------------------------------------------------
        # Save JSON
        # -------------------------------------------------------------

        meta_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}{suffix}.json'
        )

        meta = {
            'dataset': self.args.data,
            'split': flag,
            'batches': max_batches,
            'samples_per_batch': 1,
            'ranks': rank_values,
            'full_mse': full_mse,
            'full_mae': full_mae,
            'within_1_percent_ranks': within_1_percent,
            'within_3_percent_ranks': within_3_percent,
            'smallest_rank_within_1_percent': smallest_rank_1,
            'smallest_rank_within_3_percent': smallest_rank_3,
            'screen': screen
        }

        with open(
                meta_path,
                'w'
        ) as f:

            json.dump(
                meta,
                f,
                indent=2
            )

        # -------------------------------------------------------------
        # Save decision
        # -------------------------------------------------------------

        decision_path = os.path.join(
            save_dir,
            f'{self.args.data}_{flag}{suffix}_DECISION.txt'
        )

        with open(
                decision_path,
                'w'
        ) as f:

            f.write(
                'FUNCTIONAL RANK ABLATION\n'
            )

            f.write(
                '=======================\n\n'
            )

            f.write(
                f'Full-attention subset MSE: '
                f'{full_mse:.10f}\n'
            )

            f.write(
                f'Full-attention subset MAE: '
                f'{full_mae:.10f}\n\n'
            )

            f.write(
                f'Ranks within 1% MSE: '
                f'{within_1_percent}\n'
            )

            f.write(
                f'Ranks within 3% MSE: '
                f'{within_3_percent}\n'
            )

            f.write(
                f'Smallest rank within 1%: '
                f'{smallest_rank_1}\n'
            )

            f.write(
                f'Smallest rank within 3%: '
                f'{smallest_rank_3}\n\n'
            )

            f.write(
                f'Screen: {screen}\n'
            )

            f.write(
                '\nNOTE: This is a functional ablation. '
                'It does not measure runtime or memory '
                'benefits of an efficient implementation.\n'
            )

        # -------------------------------------------------------------
        # Console summary
        # -------------------------------------------------------------

        print('\n' + '=' * 90)
        print('RANK ABLATION SUMMARY')
        print('=' * 90)

        print(
            f'Full subset MSE: '
            f'{full_mse:.8f}'
        )

        print(
            f'Full subset MAE: '
            f'{full_mae:.8f}'
        )

        print(
            f'Ranks within 1%: '
            f'{within_1_percent}'
        )

        print(
            f'Ranks within 3%: '
            f'{within_3_percent}'
        )

        print(
            f'Smallest rank within 1%: '
            f'{smallest_rank_1}'
        )

        print(
            f'Smallest rank within 3%: '
            f'{smallest_rank_3}'
        )

        print(
            f'Screen: {screen}'
        )

        print('=' * 90)

        print('Saved:')
        print(csv_path)
        print(meta_path)
        print(decision_path)