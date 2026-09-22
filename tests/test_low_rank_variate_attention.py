import unittest
from types import SimpleNamespace

import torch

from layers.SelfAttention_Family import (
    FullAttention,
    InducedVariateAttention,
)
from model.iTransformer import Model


class InducedVariateAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_variable_permutation_equivariance(self):
        module = InducedVariateAttention(
            rank=3, n_heads=2, d_head=4,
            attention_dropout=0.0
        ).eval()
        q = torch.randn(2, 7, 2, 4)
        k = torch.randn(2, 7, 2, 4)
        v = torch.randn(2, 7, 2, 4)
        permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
        inverse = torch.argsort(permutation)

        expected, _ = module(q, k, v, None)
        permuted, _ = module(
            q[:, permutation], k[:, permutation], v[:, permutation], None
        )
        torch.testing.assert_close(
            expected, permuted[:, inverse], rtol=1e-5, atol=1e-6
        )

    def test_time_tokens_remain_exact_and_variable_part_is_equivariant(self):
        module = InducedVariateAttention(
            rank=2, n_heads=2, d_head=4, time_tokens=2,
            attention_dropout=0.0, output_attention=True
        ).eval()
        q = torch.randn(1, 8, 2, 4)
        k = torch.randn(1, 8, 2, 4)
        v = torch.randn(1, 8, 2, 4)
        variable_permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        full_permutation = torch.cat([
            variable_permutation, torch.tensor([6, 7])
        ])
        inverse = torch.argsort(full_permutation)

        expected, attention = module(q, k, v, None)
        permuted, _ = module(
            q[:, full_permutation],
            k[:, full_permutation],
            v[:, full_permutation],
            None
        )
        self.assertEqual(attention.shape, (1, 2, 8, 4))
        torch.testing.assert_close(
            expected, permuted[:, inverse], rtol=1e-5, atol=1e-6
        )

    def test_module_accepts_different_variable_counts(self):
        module = InducedVariateAttention(
            rank=4, n_heads=2, d_head=4,
            attention_dropout=0.0
        ).eval()
        for token_count in (5, 11):
            x = torch.randn(2, token_count, 2, 4)
            output, _ = module(x, x, x, None)
            self.assertEqual(output.shape, x.shape)

    def test_zero_gate_is_a_stable_warm_start(self):
        module = InducedVariateAttention(
            rank=3, n_heads=2, d_head=4,
            attention_dropout=0.0, gate_init=0.0
        ).eval()
        x = torch.randn(2, 7, 2, 4)
        output, _ = module(x, x, x, None)
        torch.testing.assert_close(output, torch.zeros_like(output))


class DenseOracleAttentionTest(unittest.TestCase):
    def test_capture_and_oracle_shapes(self):
        torch.manual_seed(11)
        module = FullAttention(
            mask_flag=False, attention_dropout=0.0
        ).eval()
        module.capture_diagnostics = True
        q = torch.randn(2, 6, 2, 4)
        k = torch.randn(2, 6, 2, 4)
        v = torch.randn(2, 6, 2, 4)

        dense, _ = module(q, k, v, None)
        self.assertEqual(module.last_attention.shape, (2, 2, 6, 6))
        self.assertEqual(module.last_values.shape, v.shape)

        module.oracle_rank = 1
        rank_one, _ = module(q, k, v, None)
        self.assertEqual(rank_one.shape, dense.shape)
        self.assertTrue(torch.isfinite(rank_one).all())

        module.oracle_rank = 6
        exact, _ = module(q, k, v, None)
        torch.testing.assert_close(exact, dense)


class ITransformerIntegrationTest(unittest.TestCase):
    def test_induced_attention_forward_with_time_tokens(self):
        configs = SimpleNamespace(
            seq_len=12,
            pred_len=6,
            output_attention=False,
            use_norm=True,
            embed='timeF',
            freq='h',
            dropout=0.0,
            class_strategy='projection',
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=16,
            factor=1,
            activation='gelu',
            induced_attention=True,
            attn_rank=3,
            attn_time_tokens=4,
            attn_gate_init=1.0,
        )
        model = Model(configs).eval()
        x = torch.randn(2, 12, 7)
        x_mark = torch.randn(2, 12, 4)
        output = model(x, x_mark, None, None)
        self.assertEqual(output.shape, (2, 6, 7))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
