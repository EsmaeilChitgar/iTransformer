"""Smoke and shape tests for the multi-scale patch-inverted model.

Run from the repository root with:
    python tests/test_multiscale_patch_transformer.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.MultiScalePatchEmbedding import MultiScalePatchEmbedding
from model.iMultiScalePatchTransformer import Model


def make_config(**overrides):
    config = dict(
        seq_len=37,
        pred_len=11,
        output_attention=False,
        use_norm=True,
        patch_sizes="8,16,32",
        patch_stride_ratio=0.5,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        factor=1,
        dropout=0.0,
        activation="gelu",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class MultiScalePatchTransformerTest(unittest.TestCase):
    def test_embedding_handles_covariates_and_padding(self):
        embedding = MultiScalePatchEmbedding("8,16,32", d_model=24, dropout=0.0)
        x = torch.randn(2, 37, 5)
        marks = torch.randn(2, 37, 4)
        tokens = embedding(x, marks)
        self.assertEqual(tokens.shape, (2, 9, 24))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(embedding.last_route_weights.shape, (2, 9, 3))
        self.assertTrue(torch.allclose(embedding.last_route_weights.sum(dim=-1), torch.ones(2, 9)))

    def test_model_shape_and_backward(self):
        model = Model(make_config())
        x = torch.randn(2, 37, 5, requires_grad=True)
        marks = torch.randn(2, 37, 4)
        output = model(x, marks, None, None)
        self.assertEqual(output.shape, (2, 11, 5))
        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(model.enc_embedding.scales[0].patch_projection.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_output_attention_contract(self):
        config = make_config(output_attention=True)
        model = Model(config)
        output, attention = model(torch.randn(1, 37, 3), None, None, None)
        self.assertEqual(output.shape, (1, 11, 3))
        self.assertEqual(len(attention), config.e_layers)


if __name__ == "__main__":
    unittest.main()
