"""Tests for latent cross-variate attention.

Run from the repository root with:
    python tests/test_latent_transformer.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.Latent_Variable_Attention import LatentVariableAttentionLayer
from model.iLatentTransformer import Model


def make_config(**overrides):
    config = dict(
        seq_len=24,
        pred_len=8,
        output_attention=False,
        use_norm=True,
        num_latents=4,
        latent_full_attention_threshold=8,
        d_model=16,
        n_heads=4,
        e_layers=2,
        d_ff=32,
        factor=1,
        dropout=0.0,
        activation="gelu",
        embed="timeF",
        freq="h",
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class LatentTransformerTest(unittest.TestCase):
    def test_full_attention_fallback_for_small_token_sets(self):
        layer = LatentVariableAttentionLayer(
            d_model=16, n_heads=4, num_latents=4,
            full_attention_threshold=8, output_attention=True, dropout=0.0
        )
        x = torch.randn(2, 6, 16)
        output, attention = layer(x, x, x, None)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer.last_attention_mode, "full")
        self.assertEqual(attention.shape, (2, 4, 6, 6))
        self.assertEqual(layer.attention_score_count(6), 36)

    def test_latent_path_shape_complexity_and_backward(self):
        layer = LatentVariableAttentionLayer(
            d_model=16, n_heads=4, num_latents=4,
            full_attention_threshold=8, output_attention=True, dropout=0.0
        )
        x = torch.randn(2, 20, 16, requires_grad=True)
        output, attention = layer(x, x, x, None)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer.last_attention_mode, "latent")
        self.assertEqual(attention[0].shape[0], 2)
        self.assertEqual(attention[0].shape[-2:], (4, 20))
        self.assertEqual(attention[1].shape[0], 2)
        self.assertEqual(attention[1].shape[-2:], (20, 4))
        self.assertEqual(layer.attention_score_count(20), 160)
        output.square().mean().backward()
        self.assertIsNotNone(layer.latent_tokens.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_model_supports_covariates_and_both_routes(self):
        model = Model(make_config(output_attention=True))
        small = model(torch.randn(2, 24, 5), torch.randn(2, 24, 2), None, None)
        self.assertEqual(small[0].shape, (2, 8, 5))
        self.assertEqual(model.encoder.attn_layers[0].attention.last_attention_mode, "full")

        large = model(torch.randn(2, 24, 20), torch.randn(2, 24, 2), None, None)
        self.assertEqual(large[0].shape, (2, 8, 20))
        self.assertEqual(model.encoder.attn_layers[0].attention.last_attention_mode, "latent")


if __name__ == "__main__":
    unittest.main()
