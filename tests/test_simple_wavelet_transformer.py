"""Smoke tests for the SimpleTM-inspired wavelet/geometric iTransformer."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.GeometricAttention import GeometricAttention
from layers.SimpleWaveletEmbedding import HaarStyleDecomposition
from model.iSimpleWaveletTransformer import Model


def make_config(**overrides):
    config = dict(
        seq_len=24,
        pred_len=8,
        output_attention=True,
        use_norm=True,
        wavelet_geometric_weight=0.25,
        d_model=16,
        n_heads=4,
        e_layers=2,
        d_ff=32,
        dropout=0.0,
        activation="gelu",
        embed="timeF",
        freq="h",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class SimpleWaveletTransformerTest(unittest.TestCase):
    def test_decomposition_reconstructs_pair_values(self):
        x = torch.arange(10.0).view(1, 5, 2)
        low, detail = HaarStyleDecomposition()(x)
        self.assertEqual(low.shape, x.shape)
        self.assertEqual(detail.shape, x.shape)
        reconstructed = low + detail
        self.assertTrue(torch.allclose(reconstructed[:, 0::2], x[:, 0::2]))

    def test_geometric_attention_shape_and_gradients(self):
        attention = GeometricAttention(
            output_attention=True, attention_dropout=0.0,
            geometric_weight=0.25
        )
        q = torch.randn(2, 5, 4, 4, requires_grad=True)
        output, weights = attention(q, q, q)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(weights.shape, (2, 4, 5, 5))
        self.assertTrue(torch.isfinite(weights).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_model_preserves_variates_covariates_and_gate(self):
        model = Model(make_config())
        x = torch.randn(2, 24, 7, requires_grad=True)
        marks = torch.randn(2, 24, 2)
        output, attns = model(x, marks, None, None)
        self.assertEqual(output.shape, (2, 8, 7))
        self.assertEqual(len(attns), 2)
        # Two timestamp channels are retained as the original extra tokens.
        self.assertEqual(attns[0].shape, (2, 4, 9, 9))
        self.assertEqual(model.enc_embedding.last_route_weights.shape, (2, 7, 1))
        output.square().mean().backward()
        self.assertIsNotNone(model.enc_embedding.router[1].weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
