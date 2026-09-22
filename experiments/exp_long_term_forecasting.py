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
from torch.utils.data import DataLoader

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

    def load_checkpoint(self, checkpoint_path, strict=True):
        """Load a dense or warm-start checkpoint with explicit diagnostics."""
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        state = {
            (key[7:] if key.startswith('module.') else key): value
            for key, value in state.items()
        }
        target = self.model.module if hasattr(self.model, 'module') else self.model
        incompatible = target.load_state_dict(state, strict=strict)
        if not strict:
            print('Warm-start missing keys:', incompatible.missing_keys)
            print('Warm-start unexpected keys:', incompatible.unexpected_keys)
        return incompatible

    @staticmethod
    def _energy_rank(singular_values, threshold=0.95):
        energy = singular_values.square()
        if float(energy.sum()) <= 1e-12:
            return 0
        cumulative = torch.cumsum(energy, dim=-1)
        cumulative = cumulative / cumulative[..., -1:].clamp_min(1e-12)
        return int(
            torch.searchsorted(
                cumulative, torch.tensor(
                    threshold,
                    device=cumulative.device,
                    dtype=cumulative.dtype
                )
            ).item() + 1
        )

    @staticmethod
    def _bootstrap_mean_ci(deltas, samples, seed=2023):
        values = np.asarray(deltas, dtype=np.float64)
        if values.size == 0 or samples <= 0:
            return [None, None]
        rng = np.random.default_rng(seed)
        bootstrap_means = []
        remaining = int(samples)
        while remaining:
            chunk = min(remaining, 1000)
            indices = rng.integers(
                0, values.size, size=(chunk, values.size)
            )
            bootstrap_means.append(values[indices].mean(axis=1))
            remaining -= chunk
        bootstrap_means = np.concatenate(bootstrap_means)
        return np.quantile(bootstrap_means, [0.025, 0.975]).tolist()

    def rank_analysis(self, split, ranks, max_batches=0,
                      spectral_batches=32, bootstrap_samples=10000,
                      output_dir='./rank_analysis'):
        """Evaluate SVD oracle ranks on deterministic, paired windows.

        The method reports two distinct objects:
        (1) spectral and AV errors measured on the original dense attention;
        (2) end-to-end forecasting metrics when every encoder layer is
            replaced by its per-input truncated-SVD oracle.
        It is intentionally not a runtime benchmark.
        """
        parsed_ranks = sorted({
            int(value.strip())
            for value in str(ranks).split(',')
            if value.strip() and int(value.strip()) > 0
        })
        if not parsed_ranks:
            raise ValueError('rank_analysis requires at least one rank')

        data_set, _ = self._get_data(flag=split)
        data_loader = DataLoader(
            data_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
        )
        target = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(target, 'set_oracle_rank'):
            raise RuntimeError('Selected model does not support rank analysis')

        os.makedirs(output_dir, exist_ok=True)
        conditions = [0] + parsed_ranks
        window_metrics = {
            rank: {'mse': [], 'mae': []} for rank in conditions
        }
        spectral_rows = []
        started = time.time()

        def forward_batch(batch_x, batch_x_mark, dec_inp, batch_y_mark):
            result = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark
            )
            return result[0] if isinstance(result, tuple) else result

        def record_spectral_stats(window_index):
            for layer_index, attention in enumerate(
                    target._inner_attentions(), start=1):
                A_batch = attention.last_attention
                values_batch = attention.last_values
                if A_batch is None or values_batch is None:
                    raise RuntimeError('Dense attention diagnostics were not captured')
                for sample_index in range(A_batch.shape[0]):
                    for head_index in range(A_batch.shape[1]):
                        A = A_batch[sample_index, head_index].float()
                        values = values_batch[
                            sample_index, :, head_index, :
                        ].float()
                        singular_values = torch.linalg.svdvals(A)
                        centered = A - A.mean(dim=-1, keepdim=True)
                        centered_singular_values = torch.linalg.svdvals(centered)
                        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                        dense_output = A @ values
                        row = {
                            'window': window_index,
                            'layer': layer_index,
                            'head': head_index + 1,
                            'tokens': A.shape[-1],
                            'r95': self._energy_rank(singular_values),
                            'centered_r95': self._energy_rank(
                                centered_singular_values
                            ),
                            'leading_energy': float(
                                singular_values[0].square()
                                / singular_values.square().sum().clamp_min(1e-12)
                            )
                        }
                        for rank in parsed_ranks:
                            effective_rank = min(rank, S.numel())
                            approximate_output = (
                                (U[:, :effective_rank]
                                 * S[:effective_rank].unsqueeze(0))
                                @ (Vh[:effective_rank] @ values)
                            )
                            relative_error = torch.linalg.vector_norm(
                                dense_output - approximate_output
                            ) / torch.linalg.vector_norm(
                                dense_output
                            ).clamp_min(1e-12)
                            row['av_error_r{}'.format(rank)] = float(
                                relative_error
                            )
                        spectral_rows.append(row)

        self.model.eval()
        target.set_oracle_rank(0)
        target.set_attention_diagnostics(True)
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(data_loader):
                    if max_batches and batch_index >= max_batches:
                        break
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len:, :]
                    )
                    dec_inp = torch.cat([
                        batch_y[:, :self.args.label_len, :], dec_inp
                    ], dim=1).float().to(self.device)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    truth = batch_y[:, -self.args.pred_len:, f_dim:]

                    target.set_oracle_rank(0)
                    target.set_attention_diagnostics(
                        batch_index < spectral_batches
                    )
                    dense_prediction = forward_batch(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )[:, -self.args.pred_len:, f_dim:]
                    dense_error = dense_prediction - truth
                    window_metrics[0]['mse'].append(
                        float(dense_error.square().mean())
                    )
                    window_metrics[0]['mae'].append(
                        float(dense_error.abs().mean())
                    )
                    if batch_index < spectral_batches:
                        record_spectral_stats(batch_index)

                    target.set_attention_diagnostics(False)
                    for rank in parsed_ranks:
                        target.set_oracle_rank(rank)
                        prediction = forward_batch(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[:, -self.args.pred_len:, f_dim:]
                        error = prediction - truth
                        window_metrics[rank]['mse'].append(
                            float(error.square().mean())
                        )
                        window_metrics[rank]['mae'].append(
                            float(error.abs().mean())
                        )

                    if (batch_index + 1) % 25 == 0:
                        print(
                            'Rank analysis windows: {}'.format(batch_index + 1)
                        )
        finally:
            target.set_oracle_rank(0)
            target.set_attention_diagnostics(False)

        if not window_metrics[0]['mse']:
            raise RuntimeError('rank_analysis processed no windows')

        window_path = os.path.join(output_dir, '{}_paired_windows.csv'.format(split))
        with open(window_path, 'w', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=['window', 'rank', 'mse', 'mae',
                            'mse_delta_vs_full', 'mae_delta_vs_full']
            )
            writer.writeheader()
            for rank in conditions:
                for index, mse_value in enumerate(window_metrics[rank]['mse']):
                    writer.writerow({
                        'window': index,
                        'rank': rank,
                        'mse': mse_value,
                        'mae': window_metrics[rank]['mae'][index],
                        'mse_delta_vs_full': (
                            mse_value - window_metrics[0]['mse'][index]
                        ),
                        'mae_delta_vs_full': (
                            window_metrics[rank]['mae'][index]
                            - window_metrics[0]['mae'][index]
                        )
                    })

        spectral_path = os.path.join(
            output_dir, '{}_spectral_output_errors.csv'.format(split)
        )
        if spectral_rows:
            with open(spectral_path, 'w', newline='') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(spectral_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(spectral_rows)

        full_mse = float(np.mean(window_metrics[0]['mse']))
        full_mae = float(np.mean(window_metrics[0]['mae']))
        summary = {
            'split': split,
            'windows': len(window_metrics[0]['mse']),
            'ranks': parsed_ranks,
            'spectral_windows': min(
                spectral_batches, len(window_metrics[0]['mse'])
            ),
            'full': {'mse': full_mse, 'mae': full_mae},
            'conditions': {},
            'elapsed_seconds': time.time() - started,
            'note': (
                'SVD conditions are functional oracles. They form dense '
                'attention and do not measure efficient runtime.'
            )
        }
        for rank in parsed_ranks:
            mse_values = np.asarray(window_metrics[rank]['mse'])
            mae_values = np.asarray(window_metrics[rank]['mae'])
            mse_deltas = mse_values - np.asarray(window_metrics[0]['mse'])
            mae_deltas = mae_values - np.asarray(window_metrics[0]['mae'])
            mean_mse = float(mse_values.mean())
            mean_mae = float(mae_values.mean())
            summary['conditions'][str(rank)] = {
                'mse': mean_mse,
                'mae': mean_mae,
                'mse_gap_percent_ratio_of_means': 100.0 * (
                    mean_mse - full_mse
                ) / full_mse,
                'mae_gap_percent_ratio_of_means': 100.0 * (
                    mean_mae - full_mae
                ) / full_mae,
                'mean_paired_mse_delta': float(mse_deltas.mean()),
                'mean_paired_mae_delta': float(mae_deltas.mean()),
                'paired_mse_delta_bootstrap_95': self._bootstrap_mean_ci(
                    mse_deltas, bootstrap_samples, seed=2023 + rank
                ),
                'paired_mae_delta_bootstrap_95': self._bootstrap_mean_ci(
                    mae_deltas, bootstrap_samples, seed=4046 + rank
                )
            }

        summary_path = os.path.join(
            output_dir, '{}_rank_analysis_summary.json'.format(split)
        )
        with open(summary_path, 'w') as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        print('Saved:', window_path)
        if spectral_rows:
            print('Saved:', spectral_path)
        print('Saved:', summary_path)
        return summary

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
