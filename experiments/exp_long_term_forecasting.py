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
import pandas as pd

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

        rank>0 in global mode:
            replace each attention matrix with its truncated
            rank-r SVD approximation.

        rank>0 in layer-wise mode:
            replace each attention matrix with a truncated
            SVD approximation using a different rank for each
            encoder layer.

        This experiment measures forecasting sensitivity to
        attention rank. It is NOT a runtime benchmark because
        the full attention matrix and its SVD are still computed.
        """

        model = self.model.module if hasattr(
            self.model,
            'module'
        ) else self.model

        # -------------------------------------------------------------
        # Rank configuration
        # -------------------------------------------------------------

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

        layerwise_enabled = getattr(
            self.args,
            'rank_ablation_layerwise',
            False
        )

        layer_rank_values = []

        if layerwise_enabled:

            layer_rank_text = getattr(
                self.args,
                'rank_ablation_layer_ranks',
                ''
            )

            if isinstance(
                    layer_rank_text,
                    str
            ):

                for value in layer_rank_text.split(','):

                    value = value.strip()

                    if value:
                        layer_rank_values.append(
                            int(value)
                        )

            else:

                layer_rank_values = list(
                    layer_rank_text
                )

            num_layers = len(
                model.encoder.attn_layers
            )

            if len(layer_rank_values) != num_layers:
                raise ValueError(
                    f'Expected {num_layers} layer-wise ranks, '
                    f'got {len(layer_rank_values)}: '
                    f'{layer_rank_values}'
                )

            for layer_rank in layer_rank_values:

                if layer_rank <= 0:
                    raise ValueError(
                        'Layer-wise ranks must be > 0. '
                        f'Got: {layer_rank_values}'
                    )

        # -------------------------------------------------------------
        # Output tag
        # -------------------------------------------------------------

        rank_ablation_tag = getattr(
            self.args,
            'rank_ablation_tag',
            ''
        )

        rank_ablation_tag = (
            str(
                rank_ablation_tag
            ).strip()
        )

        tag_suffix = (
            f'_{rank_ablation_tag}'
            if rank_ablation_tag
            else ''
        )

        # -------------------------------------------------------------
        # Console header
        # -------------------------------------------------------------

        print('\n' + '=' * 90)
        print('FUNCTIONAL RANK ABLATION')
        print('=' * 90)

        print(
            'Split:',
            flag
        )

        print(
            'Batches:',
            max_batches
        )

        print(
            'Ranks:',
            rank_values
        )

        print(
            'Samples per batch: 1'
        )

        print(
            'Layer-wise mode:',
            layerwise_enabled
        )

        if layerwise_enabled:

            print(
                'Layer-wise ranks:',
                layer_rank_values
            )

            print(
                'Total layer-rank budget:',
                sum(
                    layer_rank_values
                )
            )

            print(
                'Total head-rank budget:',
                sum(
                    layer_rank_values
                ) * self.args.n_heads
            )

        else:

            print(
                'Global rank mode: '
                'same rank for every layer/head'
            )

        if rank_ablation_tag:
            print(
                'Output tag:',
                rank_ablation_tag
            )

        print('=' * 90)

        # -------------------------------------------------------------
        # Data
        # -------------------------------------------------------------

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

                        # -------------------------------------------------
                        # Full attention
                        # -------------------------------------------------

                        if rank == 0:

                            model.set_rank_ablation(
                                0
                            )

                        # -------------------------------------------------
                        # Layer-wise rank ablation
                        # -------------------------------------------------

                        elif layerwise_enabled:

                            model.set_rank_ablation(
                                rank,
                                layer_ranks=layer_rank_values
                            )

                        # -------------------------------------------------
                        # Global rank ablation
                        # -------------------------------------------------

                        else:

                            model.set_rank_ablation(
                                rank
                            )

                        # -------------------------------------------------
                        # Verify the actual rank configuration.
                        # This is especially important for layer-wise
                        # ablation.
                        # -------------------------------------------------

                        if batch_idx == 0:
                            applied_ranks = [
                                int(
                                    layer.attention.inner_attention.rank_ablation
                                )
                                for layer in model.encoder.attn_layers
                            ]

                            print(
                                f'Rank {rank} applied ranks: '
                                f'{applied_ranks}'
                            )

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                        # -------------------------------------------------
                        # Compatibility with output_attention=True
                        # -------------------------------------------------

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

                    # -------------------------------------------------
                    # Always restore full attention after each batch.
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        0
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(
                        'Ablation batch '
                        f'{batch_idx + 1}/{max_batches} completed.'
                    )

        finally:

            # ---------------------------------------------------------
            # Always leave the model in normal full-attention mode.
            # ---------------------------------------------------------

            model.set_rank_ablation(
                0
            )

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

            rows.append(
                row
            )

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

        # -------------------------------------------------------------
        # Smallest rank within 1% and 3%.
        # -------------------------------------------------------------

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
            f'{tag_suffix}'
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
            writer.writerows(
                rows
            )

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
            'layerwise_enabled': layerwise_enabled,
            'layer_ranks': (
                layer_rank_values
                if layerwise_enabled
                else None
            ),
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
                f'Split: {flag}\n'
            )

            f.write(
                f'Batches: {max_batches}\n'
            )

            f.write(
                'Samples per batch: 1\n'
            )

            f.write(
                f'Layer-wise mode: '
                f'{layerwise_enabled}\n'
            )

            if layerwise_enabled:
                f.write(
                    f'Layer-wise ranks: '
                    f'{layer_rank_values}\n'
                )

                f.write(
                    f'Total layer-rank budget: '
                    f'{sum(layer_rank_values)}\n'
                )

                f.write(
                    f'Total head-rank budget: '
                    f'{sum(layer_rank_values) * self.args.n_heads}\n'
                )

            f.write('\n')

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

    def rank_ablation_paired(
            self,
            flag='val',
            max_batches=16,
            global_rank=8,
            layer_ranks='6,11,9,6'
    ):
        """
        Paired functional comparison on exactly the same validation
        windows:

            1) Full attention
            2) Global rank-R attention
            3) Layer-wise rank attention

        Each condition is evaluated on the same sample/window before
        moving to the next batch.

        This is a functional ablation only. It does not benchmark the
        runtime or memory of a true efficient low-rank implementation.
        """

        model = self.model.module if hasattr(
            self.model,
            'module'
        ) else self.model

        # -------------------------------------------------------------
        # Parse layer-wise ranks
        # -------------------------------------------------------------

        if isinstance(layer_ranks, str):

            layer_rank_values = [
                int(x.strip())
                for x in layer_ranks.split(',')
                if x.strip()
            ]

        else:

            layer_rank_values = list(
                layer_ranks
            )

        num_layers = len(
            model.encoder.attn_layers
        )

        if len(layer_rank_values) != num_layers:
            raise ValueError(
                f'Expected {num_layers} layer-wise ranks, '
                f'got {len(layer_rank_values)}: '
                f'{layer_rank_values}'
            )

        for r in layer_rank_values:

            if r <= 0:
                raise ValueError(
                    'Layer-wise ranks must be > 0. '
                    f'Got: {layer_rank_values}'
                )

        if global_rank <= 0:
            raise ValueError(
                f'global_rank must be > 0. '
                f'Got: {global_rank}'
            )

        # -------------------------------------------------------------
        # Console header
        # -------------------------------------------------------------

        print('\n' + '=' * 95)
        print('PAIRED RANK ABLATION')
        print('=' * 95)

        print(
            'Split:',
            flag
        )

        print(
            'Batches:',
            max_batches
        )

        print(
            'Samples per batch: 1'
        )

        print(
            f'Global rank: {global_rank}'
        )

        print(
            'Layer-wise ranks:',
            layer_rank_values
        )

        print(
            'Global rank budget:',
            global_rank * num_layers
        )

        print(
            'Layer-wise rank budget:',
            sum(layer_rank_values)
        )

        print('=' * 95)

        # -------------------------------------------------------------
        # Data
        # -------------------------------------------------------------

        data_set, data_loader = self._get_data(
            flag=flag
        )

        model.eval()

        # -------------------------------------------------------------
        # Per-batch paired results
        # -------------------------------------------------------------

        paired_rows = []

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
                    # Exactly one sample from the SAME DataLoader batch
                    # is used for all three conditions.
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
                    # Helper: run one condition
                    # -------------------------------------------------

                    def evaluate_current_configuration():

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

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

                        mse = torch.mean(
                            (outputs - true) ** 2
                        ).item()

                        mae = torch.mean(
                            torch.abs(outputs - true)
                        ).item()

                        return mse, mae

                    # -------------------------------------------------
                    # Condition 1: Full attention
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        0
                    )

                    if batch_idx == 0:
                        applied_ranks = [
                            int(
                                layer.attention.inner_attention.rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            f'Full applied ranks: '
                            f'{applied_ranks}'
                        )

                    full_mse, full_mae = (
                        evaluate_current_configuration()
                    )

                    # -------------------------------------------------
                    # Condition 2: Global rank
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        global_rank
                    )

                    if batch_idx == 0:
                        applied_ranks = [
                            int(
                                layer.attention.inner_attention.rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            f'Global-R{global_rank} applied ranks: '
                            f'{applied_ranks}'
                        )

                    global_mse, global_mae = (
                        evaluate_current_configuration()
                    )

                    # -------------------------------------------------
                    # Condition 3: Layer-wise rank
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        global_rank,
                        layer_ranks=layer_rank_values
                    )

                    if batch_idx == 0:
                        applied_ranks = [
                            int(
                                layer.attention.inner_attention.rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            f'LayerWise-{layer_rank_values} applied ranks: '
                            f'{applied_ranks}'
                        )

                    layerwise_mse, layerwise_mae = (
                        evaluate_current_configuration()
                    )

                    # -------------------------------------------------
                    # Paired gaps relative to the SAME Full result
                    # -------------------------------------------------

                    global_mse_gap = (
                            100.0
                            * (global_mse - full_mse)
                            / max(abs(full_mse), 1e-12)
                    )

                    global_mae_gap = (
                            100.0
                            * (global_mae - full_mae)
                            / max(abs(full_mae), 1e-12)
                    )

                    layerwise_mse_gap = (
                            100.0
                            * (layerwise_mse - full_mse)
                            / max(abs(full_mse), 1e-12)
                    )

                    layerwise_mae_gap = (
                            100.0
                            * (layerwise_mae - full_mae)
                            / max(abs(full_mae), 1e-12)
                    )

                    row = {
                        'batch': batch_idx + 1,
                        'full_mse': full_mse,
                        'global_mse': global_mse,
                        'layerwise_mse': layerwise_mse,
                        'global_mse_gap_percent': global_mse_gap,
                        'layerwise_mse_gap_percent': layerwise_mse_gap,
                        'full_mae': full_mae,
                        'global_mae': global_mae,
                        'layerwise_mae': layerwise_mae,
                        'global_mae_gap_percent': global_mae_gap,
                        'layerwise_mae_gap_percent': layerwise_mae_gap
                    }

                    paired_rows.append(
                        row
                    )

                    # -------------------------------------------------
                    # Restore full attention after each batch
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        0
                    )

                    print(
                        f'Batch {batch_idx + 1}/{max_batches}: '
                        f'Full MSE={full_mse:.8f} | '
                        f'Global-R{global_rank}={global_mse:.8f} '
                        f'({global_mse_gap:+.3f}%) | '
                        f'LayerWise={layerwise_mse:.8f} '
                        f'({layerwise_mse_gap:+.3f}%)'
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:

            model.set_rank_ablation(
                0
            )

            model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Aggregate paired results
        # -------------------------------------------------------------

        if not paired_rows:
            raise RuntimeError(
                'No paired ablation results were collected.'
            )

        full_mse_values = [
            row['full_mse']
            for row in paired_rows
        ]

        global_mse_values = [
            row['global_mse']
            for row in paired_rows
        ]

        layerwise_mse_values = [
            row['layerwise_mse']
            for row in paired_rows
        ]

        full_mae_values = [
            row['full_mae']
            for row in paired_rows
        ]

        global_mae_values = [
            row['global_mae']
            for row in paired_rows
        ]

        layerwise_mae_values = [
            row['layerwise_mae']
            for row in paired_rows
        ]

        global_mse_gaps = [
            row['global_mse_gap_percent']
            for row in paired_rows
        ]

        layerwise_mse_gaps = [
            row['layerwise_mse_gap_percent']
            for row in paired_rows
        ]

        global_mae_gaps = [
            row['global_mae_gap_percent']
            for row in paired_rows
        ]

        layerwise_mae_gaps = [
            row['layerwise_mae_gap_percent']
            for row in paired_rows
        ]

        full_mse = float(
            np.mean(full_mse_values)
        )

        global_mse = float(
            np.mean(global_mse_values)
        )

        layerwise_mse = float(
            np.mean(layerwise_mse_values)
        )

        full_mae = float(
            np.mean(full_mae_values)
        )

        global_mae = float(
            np.mean(global_mae_values)
        )

        layerwise_mae = float(
            np.mean(layerwise_mae_values)
        )

        global_mse_gap = float(
            np.mean(global_mse_gaps)
        )

        layerwise_mse_gap = float(
            np.mean(layerwise_mse_gaps)
        )

        global_mae_gap = float(
            np.mean(global_mae_gaps)
        )

        layerwise_mae_gap = float(
            np.mean(layerwise_mae_gaps)
        )

        # -------------------------------------------------------------
        # Direct comparison:
        # Layer-wise vs Global
        # -------------------------------------------------------------

        layerwise_minus_global_mse = (
                layerwise_mse - global_mse
        )

        layerwise_minus_global_mae = (
                layerwise_mae - global_mae
        )

        layerwise_vs_global_mse_percent = (
                100.0
                * layerwise_minus_global_mse
                / max(abs(global_mse), 1e-12)
        )

        layerwise_vs_global_mae_percent = (
                100.0
                * layerwise_minus_global_mae
                / max(abs(global_mae), 1e-12)
        )

        # -------------------------------------------------------------
        # Paired per-window advantage
        # -------------------------------------------------------------

        paired_mse_deltas = [
            row['layerwise_mse'] - row['global_mse']
            for row in paired_rows
        ]

        paired_mae_deltas = [
            row['layerwise_mae'] - row['global_mae']
            for row in paired_rows
        ]

        paired_mse_delta_mean = float(
            np.mean(paired_mse_deltas)
        )

        paired_mae_delta_mean = float(
            np.mean(paired_mae_deltas)
        )

        # -------------------------------------------------------------
        # Print summary
        # -------------------------------------------------------------

        print('\n' + '=' * 95)
        print('PAIRED RANK ABLATION RESULTS')
        print('=' * 95)

        print(
            f'Full: '
            f'MSE={full_mse:.10f}, '
            f'MAE={full_mae:.10f}'
        )

        print(
            f'Global-R{global_rank}: '
            f'MSE={global_mse:.10f}, '
            f'MAE={global_mae:.10f}, '
            f'MSE gap={global_mse_gap:+.4f}%, '
            f'MAE gap={global_mae_gap:+.4f}%'
        )

        print(
            f'LayerWise-{layer_rank_values}: '
            f'MSE={layerwise_mse:.10f}, '
            f'MAE={layerwise_mae:.10f}, '
            f'MSE gap={layerwise_mse_gap:+.4f}%, '
            f'MAE gap={layerwise_mae_gap:+.4f}%'
        )

        print()

        print(
            'Layer-wise vs Global-Rank:'
        )

        print(
            f'  MSE difference = '
            f'{layerwise_minus_global_mse:+.10f}'
        )

        print(
            f'  MSE relative = '
            f'{layerwise_vs_global_mse_percent:+.4f}%'
        )

        print(
            f'  MAE difference = '
            f'{layerwise_minus_global_mae:+.10f}'
        )

        print(
            f'  MAE relative = '
            f'{layerwise_vs_global_mae_percent:+.4f}%'
        )

        print(
            f'  Mean paired MSE delta = '
            f'{paired_mse_delta_mean:+.10f}'
        )

        print(
            f'  Mean paired MAE delta = '
            f'{paired_mae_delta_mean:+.10f}'
        )

        print('=' * 95)

        # -------------------------------------------------------------
        # Save files
        # -------------------------------------------------------------

        save_dir = './rank_diagnostic'

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        rank_string = '_'.join(
            str(x)
            for x in layer_rank_values
        )

        base_name = (
            f'{self.args.data}_{flag}'
            f'_paired_g{global_rank}'
            f'_lw_{rank_string}'
            f'_b{max_batches}'
        )

        paired_csv_path = os.path.join(
            save_dir,
            f'{base_name}.csv'
        )

        with open(
                paired_csv_path,
                'w',
                newline=''
        ) as f:

            fieldnames = [
                'batch',
                'full_mse',
                'global_mse',
                'layerwise_mse',
                'global_mse_gap_percent',
                'layerwise_mse_gap_percent',
                'full_mae',
                'global_mae',
                'layerwise_mae',
                'global_mae_gap_percent',
                'layerwise_mae_gap_percent'
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                paired_rows
            )

        # -------------------------------------------------------------
        # Summary JSON
        # -------------------------------------------------------------

        json_path = os.path.join(
            save_dir,
            f'{base_name}.json'
        )

        summary = {
            'dataset': self.args.data,
            'split': flag,
            'batches': len(paired_rows),
            'samples_per_batch': 1,
            'global_rank': global_rank,
            'global_rank_budget': global_rank * num_layers,
            'layer_ranks': layer_rank_values,
            'layerwise_rank_budget': sum(layer_rank_values),
            'full_mse': full_mse,
            'global_mse': global_mse,
            'layerwise_mse': layerwise_mse,
            'global_mse_gap_percent_vs_full': global_mse_gap,
            'layerwise_mse_gap_percent_vs_full': layerwise_mse_gap,
            'full_mae': full_mae,
            'global_mae': global_mae,
            'layerwise_mae': layerwise_mae,
            'global_mae_gap_percent_vs_full': global_mae_gap,
            'layerwise_mae_gap_percent_vs_full': layerwise_mae_gap,
            'layerwise_minus_global_mse': layerwise_minus_global_mse,
            'layerwise_vs_global_mse_percent': layerwise_vs_global_mse_percent,
            'layerwise_minus_global_mae': layerwise_minus_global_mae,
            'layerwise_vs_global_mae_percent': layerwise_vs_global_mae_percent,
            'mean_paired_mse_delta': paired_mse_delta_mean,
            'mean_paired_mae_delta': paired_mae_delta_mean,
            'note': (
                'All three conditions were evaluated on the exact same '
                'validation windows within each batch.'
            )
        }

        with open(
                json_path,
                'w'
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        # -------------------------------------------------------------
        # Decision file
        # -------------------------------------------------------------

        decision_path = os.path.join(
            save_dir,
            f'{base_name}_DECISION.txt'
        )

        with open(
                decision_path,
                'w'
        ) as f:

            f.write(
                'PAIRED GLOBAL-RANK VS LAYER-WISE ABLATION\n'
            )

            f.write(
                '=========================================\n\n'
            )

            f.write(
                f'Split: {flag}\n'
            )

            f.write(
                f'Batches: {len(paired_rows)}\n'
            )

            f.write(
                'Samples per batch: 1\n\n'
            )

            f.write(
                f'Global rank: {global_rank}\n'
            )

            f.write(
                f'Global rank budget: '
                f'{global_rank * num_layers}\n'
            )

            f.write(
                f'Layer-wise ranks: '
                f'{layer_rank_values}\n'
            )

            f.write(
                f'Layer-wise rank budget: '
                f'{sum(layer_rank_values)}\n\n'
            )

            f.write(
                f'Full MSE: {full_mse:.10f}\n'
            )

            f.write(
                f'Global-R{global_rank} MSE: '
                f'{global_mse:.10f}\n'
            )

            f.write(
                f'Layer-wise MSE: '
                f'{layerwise_mse:.10f}\n\n'
            )

            f.write(
                f'Global MSE gap vs Full: '
                f'{global_mse_gap:+.6f}%\n'
            )

            f.write(
                f'Layer-wise MSE gap vs Full: '
                f'{layerwise_mse_gap:+.6f}%\n\n'
            )

            f.write(
                f'Layer-wise vs Global MSE difference: '
                f'{layerwise_minus_global_mse:+.10f}\n'
            )

            f.write(
                f'Layer-wise vs Global MSE relative: '
                f'{layerwise_vs_global_mse_percent:+.6f}%\n\n'
            )

            f.write(
                f'Full MAE: {full_mae:.10f}\n'
            )

            f.write(
                f'Global-R{global_rank} MAE: '
                f'{global_mae:.10f}\n'
            )

            f.write(
                f'Layer-wise MAE: '
                f'{layerwise_mae:.10f}\n\n'
            )

            f.write(
                f'Global MAE gap vs Full: '
                f'{global_mae_gap:+.6f}%\n'
            )

            f.write(
                f'Layer-wise MAE gap vs Full: '
                f'{layerwise_mae_gap:+.6f}%\n\n'
            )

            f.write(
                f'Layer-wise vs Global MAE difference: '
                f'{layerwise_minus_global_mae:+.10f}\n'
            )

            f.write(
                f'Layer-wise vs Global MAE relative: '
                f'{layerwise_vs_global_mae_percent:+.6f}%\n\n'
            )

            f.write(
                'NOTE:\n'
            )

            f.write(
                'This is a paired functional ablation. '
                'The same validation windows are used for Full, '
                'Global-Rank, and Layer-wise conditions. '
                'It does not measure runtime or memory benefits '
                'of a true efficient low-rank implementation.\n'
            )

        print()
        print('Saved:')
        print(paired_csv_path)
        print(json_path)
        print(decision_path)

    def rank_ablation_paired_headwise(
            self,
            flag='val',
            max_batches=16,
            global_rank=8,
            attention_summary_csv=(
                    './rank_diagnostic/'
                    'custom_val_attention_summary_b32.csv'
            )
    ):
        """
        Paired functional comparison on exactly the same windows:

            1) Full attention
            2) Global-R8
            3) Head-wise rank allocation

        The head-wise allocation is derived from the previously
        computed per-head r95 values in attention_summary_csv.

        The total head-wise rank budget is forced to equal:

            num_layers * num_heads * global_rank

        For the current Traffic setup:

            4 * 8 * 8 = 256

        IMPORTANT:
        This is a functional/oracle ablation.

        The implementation still computes:

            full attention matrix
            full SVD

        Therefore it does NOT measure the runtime/memory benefit
        of a true low-rank attention implementation.
        """

        # =============================================================
        # Helpers
        # =============================================================

        def _find_column(
                dataframe,
                candidates
        ):

            normalized = {
                str(column).strip().lower(): column
                for column in dataframe.columns
            }

            for candidate in candidates:

                key = candidate.strip().lower()

                if key in normalized:
                    return normalized[key]

            # Fuzzy fallback
            for column in dataframe.columns:

                normalized_column = (
                    str(column)
                    .strip()
                    .lower()
                    .replace('_', '')
                    .replace('-', '')
                    .replace(' ', '')
                )

                for candidate in candidates:

                    normalized_candidate = (
                        candidate
                        .strip()
                        .lower()
                        .replace('_', '')
                        .replace('-', '')
                        .replace(' ', '')
                    )

                    if normalized_column == normalized_candidate:
                        return column

            return None

        def _parse_index(value):

            if isinstance(
                    value,
                    (int, np.integer)
            ):
                return int(value)

            if isinstance(
                    value,
                    (float, np.floating)
            ):
                return int(value)

            text_value = str(
                value
            ).strip()

            # Examples:
            # L1 -> 1
            # Layer2 -> 2
            # H3 -> 3
            # Head8 -> 8

            digits = ''.join(
                character
                for character in text_value
                if character.isdigit()
            )

            if not digits:
                raise ValueError(
                    f'Cannot parse layer/head index from: '
                    f'{value}'
                )

            return int(digits)

        def _allocate_integer_budget(
                weights,
                budget,
                min_rank=1
        ):
            """
            Convert positive continuous weights into integer ranks
            with an exact total budget.

            Uses proportional allocation + largest remainder.

            Returns:
                np.ndarray[int]
            """

            weights = np.asarray(
                weights,
                dtype=np.float64
            )

            if np.any(
                    ~np.isfinite(weights)
            ):
                raise ValueError(
                    'Non-finite values found in rank weights.'
                )

            if np.any(
                    weights <= 0
            ):
                raise ValueError(
                    f'Rank weights must be > 0. '
                    f'Got: {weights.tolist()}'
                )

            num_items = len(
                weights
            )

            if budget < (
                    num_items * min_rank
            ):
                raise ValueError(
                    f'Budget {budget} is too small for '
                    f'{num_items} items with min_rank={min_rank}.'
                )

            ideal = (
                    weights
                    / weights.sum()
                    * budget
            )

            ranks = np.floor(
                ideal
            ).astype(
                np.int64
            )

            ranks = np.maximum(
                ranks,
                min_rank
            )

            current_sum = int(
                ranks.sum()
            )

            # ---------------------------------------------------------
            # Add remaining units
            # ---------------------------------------------------------

            while current_sum < budget:

                fractions = (
                        ideal - ranks
                )

                order = np.argsort(
                    -fractions
                )

                changed = False

                for idx in order:

                    if current_sum >= budget:
                        break

                    ranks[idx] += 1
                    current_sum += 1
                    changed = True

                if not changed:
                    break

            # ---------------------------------------------------------
            # Remove excess units if minimum-rank protection caused it.
            # ---------------------------------------------------------

            while current_sum > budget:

                candidates = np.where(
                    ranks > min_rank
                )[0]

                if len(candidates) == 0:
                    raise RuntimeError(
                        'Unable to satisfy exact rank budget.'
                    )

                # Remove first from the rank with the smallest
                # fractional remainder.
                deficits = (
                        ranks[candidates]
                        - ideal[candidates]
                )

                order = candidates[
                    np.argsort(
                        -deficits
                    )
                ]

                changed = False

                for idx in order:

                    if current_sum <= budget:
                        break

                    if ranks[idx] > min_rank:
                        ranks[idx] -= 1
                        current_sum -= 1
                        changed = True

                if not changed:
                    raise RuntimeError(
                        'Unable to reduce rank allocation.'
                    )

            return ranks.astype(
                np.int64
            )

        # =============================================================
        # Basic model information
        # =============================================================

        model = (
            self.model.module
            if hasattr(
                self.model,
                'module'
            )
            else self.model
        )

        num_layers = len(
            model.encoder.attn_layers
        )

        num_heads = int(
            self.args.n_heads
        )

        total_budget = (
                num_layers
                * num_heads
                * global_rank
        )

        # =============================================================
        # Load attention summary CSV
        # =============================================================

        if not os.path.exists(
                attention_summary_csv
        ):
            raise FileNotFoundError(
                'Attention summary CSV not found:\n'
                f'{attention_summary_csv}'
            )

        summary_df = pd.read_csv(
            attention_summary_csv
        )

        # -------------------------------------------------------------
        # Find columns
        # -------------------------------------------------------------

        layer_column = _find_column(
            summary_df,
            [
                'layer',
                'layer_id',
                'layer_idx'
            ]
        )

        head_column = _find_column(
            summary_df,
            [
                'head',
                'head_id',
                'head_idx'
            ]
        )

        r95_column = _find_column(
            summary_df,
            [
                'r95',
                'rank95',
                'rank_95',
                'r_95',
                'r95_rank',
                'effective_rank_95'
            ]
        )

        if layer_column is None:
            raise ValueError(
                'Could not find a layer column in:\n'
                f'{summary_df.columns.tolist()}'
            )

        if head_column is None:
            raise ValueError(
                'Could not find a head column in:\n'
                f'{summary_df.columns.tolist()}'
            )

        if r95_column is None:
            raise ValueError(
                'Could not find an r95 column in:\n'
                f'{summary_df.columns.tolist()}'
            )

        # =============================================================
        # Build lookup
        # =============================================================

        head_weight_lookup = {}

        for _, row in summary_df.iterrows():
            layer_id = _parse_index(
                row[layer_column]
            )

            head_id = _parse_index(
                row[head_column]
            )

            r95_value = float(
                row[r95_column]
            )

            head_weight_lookup[
                (layer_id, head_id)
            ] = r95_value

        # -------------------------------------------------------------
        # Detect 1-based indexing
        # -------------------------------------------------------------

        layer_ids = [
            key[0]
            for key in head_weight_lookup
        ]

        head_ids = [
            key[1]
            for key in head_weight_lookup
        ]

        if (
                min(layer_ids) >= 1
                and max(layer_ids) <= num_layers
        ):

            layer_offset = 1

        else:

            layer_offset = 0

        if (
                min(head_ids) >= 1
                and max(head_ids) <= num_heads
        ):

            head_offset = 1

        else:

            head_offset = 0

        # =============================================================
        # Create ordered weight matrix
        # =============================================================

        weight_matrix = np.zeros(
            (
                num_layers,
                num_heads
            ),
            dtype=np.float64
        )

        for layer_idx in range(
                num_layers
        ):

            for head_idx in range(
                    num_heads
            ):

                source_key = (
                    layer_idx + layer_offset,
                    head_idx + head_offset
                )

                if source_key not in head_weight_lookup:
                    raise ValueError(
                        'Missing r95 entry for '
                        f'layer={layer_idx + 1}, '
                        f'head={head_idx + 1}'
                    )

                weight_matrix[
                    layer_idx,
                    head_idx
                ] = head_weight_lookup[
                    source_key
                ]

        # =============================================================
        # Convert all 32 r95 values into an exact 256-unit budget
        # =============================================================

        flat_weights = (
            weight_matrix.reshape(-1)
        )

        flat_headwise_ranks = (
            _allocate_integer_budget(
                flat_weights,
                total_budget,
                min_rank=1
            )
        )

        headwise_rank_matrix = (
            flat_headwise_ranks.reshape(
                num_layers,
                num_heads
            )
        )

        # =============================================================
        # Console header
        # =============================================================

        print('\n' + '=' * 100)
        print('PAIRED GLOBAL-RANK VS HEAD-WISE RANK ABLATION')
        print('=' * 100)

        print(
            'Split:',
            flag
        )

        print(
            'Batches:',
            max_batches
        )

        print(
            'Samples per batch: 1'
        )

        print(
            'Global rank:',
            global_rank
        )

        print(
            'Number of layers:',
            num_layers
        )

        print(
            'Number of heads:',
            num_heads
        )

        print(
            'Global rank budget:',
            total_budget
        )

        print(
            'Head-wise rank budget:',
            int(
                headwise_rank_matrix.sum()
            )
        )

        print(
            'Source CSV:',
            attention_summary_csv
        )

        print(
            'r95 column:',
            r95_column
        )

        print('\nHead-wise rank allocation:')

        for layer_idx in range(
                num_layers
        ):
            print(
                f'Layer {layer_idx + 1}: '
                f'{headwise_rank_matrix[layer_idx].tolist()}'
            )

        print(
            '\nTotal head-wise budget:',
            int(
                headwise_rank_matrix.sum()
            )
        )

        if int(
                headwise_rank_matrix.sum()
        ) != total_budget:
            raise RuntimeError(
                'Head-wise rank budget mismatch.'
            )

        print('=' * 100)

        # =============================================================
        # Data
        # =============================================================

        data_set, data_loader = self._get_data(
            flag=flag
        )

        model.eval()

        paired_rows = []

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
                    # Same exact sample/window for all 3 conditions
                    # -------------------------------------------------

                    batch_x = batch_x[
                              :1
                              ].float().to(
                        self.device
                    )

                    batch_y = batch_y[
                              :1
                              ].float().to(
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
                            batch_x_mark[
                            :1
                            ]
                            .float()
                            .to(
                                self.device
                            )
                        )

                        batch_y_mark = (
                            batch_y_mark[
                            :1
                            ]
                            .float()
                            .to(
                                self.device
                            )
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
                    ).float().to(
                        self.device
                    )

                    # =================================================
                    # Evaluation helper
                    # =================================================

                    def evaluate_current_configuration():

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

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

                        mse = torch.mean(
                            (outputs - true) ** 2
                        ).item()

                        mae = torch.mean(
                            torch.abs(outputs - true)
                        ).item()

                        return mse, mae

                    # =================================================
                    # 1) FULL
                    # =================================================

                    model.set_rank_ablation(
                        0
                    )

                    if batch_idx == 0:
                        applied_full = [
                            (
                                layer
                                .attention
                                .inner_attention
                                .rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            '\nFULL applied ranks:',
                            applied_full
                        )

                    full_mse, full_mae = (
                        evaluate_current_configuration()
                    )

                    # =================================================
                    # 2) GLOBAL R=8
                    # =================================================

                    model.set_rank_ablation(
                        global_rank
                    )

                    if batch_idx == 0:
                        applied_global = [
                            int(
                                layer
                                .attention
                                .inner_attention
                                .rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            'Global-R8 applied ranks:',
                            applied_global
                        )

                    global_mse, global_mae = (
                        evaluate_current_configuration()
                    )

                    # =================================================
                    # 3) HEAD-WISE
                    # =================================================

                    model.set_rank_ablation(
                        global_rank,
                        head_ranks=[
                            row.tolist()
                            for row in headwise_rank_matrix
                        ]
                    )

                    if batch_idx == 0:

                        applied_headwise = [
                            list(
                                layer
                                .attention
                                .inner_attention
                                .head_rank_ablation
                            )
                            for layer in model.encoder.attn_layers
                        ]

                        print(
                            'HeadWise applied ranks:'
                        )

                        for layer_idx, ranks_for_layer in enumerate(
                                applied_headwise
                        ):
                            print(
                                f'  Layer {layer_idx + 1}: '
                                f'{ranks_for_layer}'
                            )

                    headwise_mse, headwise_mae = (
                        evaluate_current_configuration()
                    )

                    # =================================================
                    # Paired gaps relative to SAME full window
                    # =================================================

                    global_mse_gap = (
                            100.0
                            * (
                                    global_mse
                                    - full_mse
                            )
                            / max(
                        abs(full_mse),
                        1e-12
                    )
                    )

                    headwise_mse_gap = (
                            100.0
                            * (
                                    headwise_mse
                                    - full_mse
                            )
                            / max(
                        abs(full_mse),
                        1e-12
                    )
                    )

                    global_mae_gap = (
                            100.0
                            * (
                                    global_mae
                                    - full_mae
                            )
                            / max(
                        abs(full_mae),
                        1e-12
                    )
                    )

                    headwise_mae_gap = (
                            100.0
                            * (
                                    headwise_mae
                                    - full_mae
                            )
                            / max(
                        abs(full_mae),
                        1e-12
                    )
                    )

                    headwise_minus_global_mse = (
                            headwise_mse
                            - global_mse
                    )

                    headwise_minus_global_mae = (
                            headwise_mae
                            - global_mae
                    )

                    row = {
                        'batch': batch_idx + 1,

                        'full_mse': full_mse,
                        'global_mse': global_mse,
                        'headwise_mse': headwise_mse,

                        'global_mse_gap_percent': global_mse_gap,
                        'headwise_mse_gap_percent': headwise_mse_gap,

                        'full_mae': full_mae,
                        'global_mae': global_mae,
                        'headwise_mae': headwise_mae,

                        'global_mae_gap_percent': global_mae_gap,
                        'headwise_mae_gap_percent': headwise_mae_gap,

                        'headwise_minus_global_mse':
                            headwise_minus_global_mse,

                        'headwise_minus_global_mae':
                            headwise_minus_global_mae
                    }

                    paired_rows.append(
                        row
                    )

                    # -------------------------------------------------
                    # Restore full attention
                    # -------------------------------------------------

                    model.set_rank_ablation(
                        0
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(
                        f'Batch {batch_idx + 1}/{max_batches}: '
                        f'Full={full_mse:.8f} | '
                        f'Global-R{global_rank}='
                        f'{global_mse:.8f} '
                        f'({global_mse_gap:+.3f}%) | '
                        f'HeadWise='
                        f'{headwise_mse:.8f} '
                        f'({headwise_mse_gap:+.3f}%)'
                    )

        finally:

            model.set_rank_ablation(
                0
            )

            model.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =============================================================
        # Validate collected results
        # =============================================================

        if not paired_rows:
            raise RuntimeError(
                'No paired head-wise results were collected.'
            )

        # =============================================================
        # Aggregate
        # =============================================================

        full_mse = float(
            np.mean([
                row['full_mse']
                for row in paired_rows
            ])
        )

        global_mse = float(
            np.mean([
                row['global_mse']
                for row in paired_rows
            ])
        )

        headwise_mse = float(
            np.mean([
                row['headwise_mse']
                for row in paired_rows
            ])
        )

        full_mae = float(
            np.mean([
                row['full_mae']
                for row in paired_rows
            ])
        )

        global_mae = float(
            np.mean([
                row['global_mae']
                for row in paired_rows
            ])
        )

        headwise_mae = float(
            np.mean([
                row['headwise_mae']
                for row in paired_rows
            ])
        )

        global_mse_gap = float(
            np.mean([
                row['global_mse_gap_percent']
                for row in paired_rows
            ])
        )

        headwise_mse_gap = float(
            np.mean([
                row['headwise_mse_gap_percent']
                for row in paired_rows
            ])
        )

        global_mae_gap = float(
            np.mean([
                row['global_mae_gap_percent']
                for row in paired_rows
            ])
        )

        headwise_mae_gap = float(
            np.mean([
                row['headwise_mae_gap_percent']
                for row in paired_rows
            ])
        )

        headwise_minus_global_mse = (
                headwise_mse
                - global_mse
        )

        headwise_minus_global_mae = (
                headwise_mae
                - global_mae
        )

        headwise_vs_global_mse_percent = (
                100.0
                * headwise_minus_global_mse
                / max(
            abs(global_mse),
            1e-12
        )
        )

        headwise_vs_global_mae_percent = (
                100.0
                * headwise_minus_global_mae
                / max(
            abs(global_mae),
            1e-12
        )
        )

        paired_mse_deltas = [
            row['headwise_minus_global_mse']
            for row in paired_rows
        ]

        paired_mae_deltas = [
            row['headwise_minus_global_mae']
            for row in paired_rows
        ]

        mean_paired_mse_delta = float(
            np.mean(
                paired_mse_deltas
            )
        )

        mean_paired_mae_delta = float(
            np.mean(
                paired_mae_deltas
            )
        )

        headwise_mse_wins = sum(
            1
            for delta in paired_mse_deltas
            if delta < 0
        )

        headwise_mse_losses = sum(
            1
            for delta in paired_mse_deltas
            if delta > 0
        )

        headwise_mse_ties = sum(
            1
            for delta in paired_mse_deltas
            if delta == 0
        )

        # =============================================================
        # Print final result
        # =============================================================

        print('\n' + '=' * 100)
        print('PAIRED HEAD-WISE RANK ABLATION RESULTS')
        print('=' * 100)

        print(
            f'Full: '
            f'MSE={full_mse:.10f}, '
            f'MAE={full_mae:.10f}'
        )

        print(
            f'Global-R{global_rank}: '
            f'MSE={global_mse:.10f}, '
            f'MAE={global_mae:.10f}, '
            f'MSE gap={global_mse_gap:+.4f}%, '
            f'MAE gap={global_mae_gap:+.4f}%'
        )

        print(
            f'HeadWise: '
            f'MSE={headwise_mse:.10f}, '
            f'MAE={headwise_mae:.10f}, '
            f'MSE gap={headwise_mse_gap:+.4f}%, '
            f'MAE gap={headwise_mae_gap:+.4f}%'
        )

        print()

        print(
            'HeadWise vs Global-R8:'
        )

        print(
            f'  MSE difference = '
            f'{headwise_minus_global_mse:+.10f}'
        )

        print(
            f'  MSE relative = '
            f'{headwise_vs_global_mse_percent:+.4f}%'
        )

        print(
            f'  MAE difference = '
            f'{headwise_minus_global_mae:+.10f}'
        )

        print(
            f'  MAE relative = '
            f'{headwise_vs_global_mae_percent:+.4f}%'
        )

        print()

        print(
            'Paired MSE windows:'
        )

        print(
            f'  HeadWise wins  : '
            f'{headwise_mse_wins}'
        )

        print(
            f'  HeadWise losses: '
            f'{headwise_mse_losses}'
        )

        print(
            f'  Ties           : '
            f'{headwise_mse_ties}'
        )

        print(
            f'  Mean paired MSE delta: '
            f'{mean_paired_mse_delta:+.10f}'
        )

        print(
            f'  Mean paired MAE delta: '
            f'{mean_paired_mae_delta:+.10f}'
        )

        print('=' * 100)

        # =============================================================
        # Save
        # =============================================================

        save_dir = './rank_diagnostic'

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        schedule_string = '_'.join(
            str(int(x))
            for x in headwise_rank_matrix.reshape(-1)
        )

        base_name = (
            f'{self.args.data}_{flag}'
            f'_paired_g{global_rank}'
            f'_headwise'
            f'_b{max_batches}'
        )

        csv_path = os.path.join(
            save_dir,
            f'{base_name}.csv'
        )

        fieldnames = list(
            paired_rows[0].keys()
        )

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

            writer.writerows(
                paired_rows
            )

        # =============================================================
        # Save JSON
        # =============================================================

        json_path = os.path.join(
            save_dir,
            f'{base_name}.json'
        )

        summary = {
            'dataset': self.args.data,
            'split': flag,
            'batches': len(paired_rows),
            'samples_per_batch': 1,

            'global_rank': global_rank,
            'global_rank_budget': total_budget,

            'headwise_rank_budget': int(
                headwise_rank_matrix.sum()
            ),

            'headwise_rank_matrix': (
                headwise_rank_matrix.tolist()
            ),

            'r95_source_csv': (
                attention_summary_csv
            ),

            'r95_column': r95_column,

            'r95_weight_matrix': (
                weight_matrix.tolist()
            ),

            'full_mse': full_mse,
            'global_mse': global_mse,
            'headwise_mse': headwise_mse,

            'global_mse_gap_percent_vs_full':
                global_mse_gap,

            'headwise_mse_gap_percent_vs_full':
                headwise_mse_gap,

            'full_mae': full_mae,
            'global_mae': global_mae,
            'headwise_mae': headwise_mae,

            'global_mae_gap_percent_vs_full':
                global_mae_gap,

            'headwise_mae_gap_percent_vs_full':
                headwise_mae_gap,

            'headwise_minus_global_mse':
                headwise_minus_global_mse,

            'headwise_vs_global_mse_percent':
                headwise_vs_global_mse_percent,

            'headwise_minus_global_mae':
                headwise_minus_global_mae,

            'headwise_vs_global_mae_percent':
                headwise_vs_global_mae_percent,

            'headwise_mse_wins':
                headwise_mse_wins,

            'headwise_mse_losses':
                headwise_mse_losses,

            'headwise_mse_ties':
                headwise_mse_ties,

            'mean_paired_mse_delta':
                mean_paired_mse_delta,

            'mean_paired_mae_delta':
                mean_paired_mae_delta,

            'note': (
                'Full, Global-Rank, and Head-wise conditions '
                'were evaluated on the exact same validation '
                'windows. The Head-wise schedule is derived '
                'from r95 weights and normalized to the exact '
                'same total rank budget as Global-Rank.'
            )
        }

        with open(
                json_path,
                'w'
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        # =============================================================
        # Decision file
        # =============================================================

        decision_path = os.path.join(
            save_dir,
            f'{base_name}_DECISION.txt'
        )

        with open(
                decision_path,
                'w'
        ) as f:

            f.write(
                'PAIRED GLOBAL-R8 VS HEAD-WISE RANK ABLATION\n'
            )

            f.write(
                '============================================\n\n'
            )

            f.write(
                f'Split: {flag}\n'
            )

            f.write(
                f'Batches: {len(paired_rows)}\n'
            )

            f.write(
                'Samples per batch: 1\n\n'
            )

            f.write(
                f'Global rank: {global_rank}\n'
            )

            f.write(
                f'Global rank budget: '
                f'{total_budget}\n'
            )

            f.write(
                f'Head-wise rank budget: '
                f'{int(headwise_rank_matrix.sum())}\n\n'
            )

            f.write(
                'Head-wise rank matrix:\n'
            )

            for layer_idx in range(
                    num_layers
            ):
                f.write(
                    f'Layer {layer_idx + 1}: '
                    f'{headwise_rank_matrix[layer_idx].tolist()}\n'
                )

            f.write('\n')

            f.write(
                f'Full MSE: '
                f'{full_mse:.10f}\n'
            )

            f.write(
                f'Global-R8 MSE: '
                f'{global_mse:.10f}\n'
            )

            f.write(
                f'Head-wise MSE: '
                f'{headwise_mse:.10f}\n\n'
            )

            f.write(
                f'Global MSE gap vs Full: '
                f'{global_mse_gap:+.6f}%\n'
            )

            f.write(
                f'Head-wise MSE gap vs Full: '
                f'{headwise_mse_gap:+.6f}%\n\n'
            )

            f.write(
                f'Head-wise vs Global MSE difference: '
                f'{headwise_minus_global_mse:+.10f}\n'
            )

            f.write(
                f'Head-wise vs Global MSE relative: '
                f'{headwise_vs_global_mse_percent:+.6f}%\n\n'
            )

            f.write(
                f'Full MAE: '
                f'{full_mae:.10f}\n'
            )

            f.write(
                f'Global-R8 MAE: '
                f'{global_mae:.10f}\n'
            )

            f.write(
                f'Head-wise MAE: '
                f'{headwise_mae:.10f}\n\n'
            )

            f.write(
                f'Global MAE gap vs Full: '
                f'{global_mae_gap:+.6f}%\n'
            )

            f.write(
                f'Head-wise MAE gap vs Full: '
                f'{headwise_mae_gap:+.6f}%\n\n'
            )

            f.write(
                f'Head-wise MSE wins: '
                f'{headwise_mse_wins}\n'
            )

            f.write(
                f'Head-wise MSE losses: '
                f'{headwise_mse_losses}\n'
            )

            f.write(
                f'Head-wise MSE ties: '
                f'{headwise_mse_ties}\n\n'
            )

            f.write(
                'NOTE:\n'
            )

            f.write(
                'This is a paired functional/oracle ablation. '
                'The exact same validation windows are used for '
                'Full, Global-R8, and Head-wise conditions. '
                'The implementation still computes full attention '
                'and full SVD, so runtime/memory improvement of a '
                'true efficient low-rank implementation is NOT '
                'measured here.\n'
            )

        print()
        print('Saved:')
        print(csv_path)
        print(json_path)
        print(decision_path)