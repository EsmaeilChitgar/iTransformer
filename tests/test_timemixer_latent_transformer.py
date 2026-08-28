"""Tests for the TimeMixer + latent iTransformer hybrid."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.TemporalScaleMixer import TemporalScaleMixer
from model.iTimeMixerLatentTransformer import Model


def make_config(**overrides):
    config = dict(
        seq_len=24,
        pred_len=8,
        output_attention=False,
        use_norm=True,
        temporal_scales="3,7,15",
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


class TimeMixerLatentTest(unittest.TestCase):
    def test_temporal_mixer_preserves_shape_and_routes(self):
        mixer = TemporalScaleMixer("3,7,15")
        x = torch.randn(2, 24, 5)
        mixed = mixer(x)
        self.assertEqual(mixed.shape, x.shape)
        self.assertEqual(mixer.last_route_weights.shape, (2, 5, 7))
        self.assertTrue(torch.allclose(
            mixer.last_route_weights.sum(dim=-1), torch.ones(2, 5)
        ))

    def test_hybrid_uses_latents_for_large_inputs(self):
        model = Model(make_config())
        x = torch.randn(2, 24, 20, requires_grad=True)
        marks = torch.randn(2, 24, 2)
        output = model(x, marks, None, None)
        self.assertEqual(output.shape, (2, 8, 20))
        attention = model.encoder.attn_layers[0].attention
        self.assertEqual(attention.last_attention_mode, "latent")
        output.square().mean().backward()
        self.assertIsNotNone(model.temporal_mixer.router[1].weight.grad)
        self.assertIsNotNone(attention.latent_tokens.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_hybrid_falls_back_for_small_inputs(self):
        model = Model(make_config())
        output = model(torch.randn(2, 24, 5), None, None, None)
        self.assertEqual(output.shape, (2, 8, 5))
        self.assertEqual(
            model.encoder.attn_layers[0].attention.last_attention_mode, "full"
        )


if __name__ == "__main__":
    unittest.main()
