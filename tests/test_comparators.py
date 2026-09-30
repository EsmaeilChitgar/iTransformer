"""Standalone CPU structural tests for paper-derived research baselines.

Usage (from iTransformer repo root): python tests/test_comparators.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import math
import torch
from torch import nn
from torch.nn import functional as F
from layers.ILRA_Comparators import (SetTransformerMAB, ISABEncoderLayer,
                                     LunaEncoder, LunaEncoderLayer)


class LiteralOfficialMAB(nn.Module):
    """Independent reference construction of juho-lee/set_transformer MAB."""
    def __init__(self, dim_v=16, num_heads=4):
        super().__init__()
        self.dim_V, self.num_heads = dim_v, num_heads
        self.fc_q = nn.Linear(dim_v, dim_v)
        self.fc_k = nn.Linear(dim_v, dim_v)
        self.fc_v = nn.Linear(dim_v, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, q, k):
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)
        dh = self.dim_V // self.num_heads
        qh = torch.cat(q.split(dh, 2), 0)
        kh = torch.cat(k.split(dh, 2), 0)
        vh = torch.cat(v.split(dh, 2), 0)
        a = torch.softmax(qh.bmm(kh.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        out = torch.cat((qh + a.bmm(vh)).split(q.size(0), 0), 2)
        return out + F.relu(self.fc_o(out))


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def test_mab_exact_match_reference_weights(self):
        reference = LiteralOfficialMAB()
        implementation = SetTransformerMAB(16, 16, 16, 4, ln=False)
        implementation.load_state_dict(reference.state_dict())
        q, k = torch.randn(2, 7, 16), torch.randn(2, 11, 16)
        torch.testing.assert_close(implementation(q, k), reference(q, k),
                                   rtol=1e-5, atol=1e-6)

    def test_isab_variable_permutation_equivariance(self):
        isab = ISABEncoderLayer(16, 4, 3, layernorm=False).eval()
        x = torch.randn(2, 10, 16)
        perm = torch.tensor([5, 1, 3, 4, 2, 0, 7, 8, 9, 6])
        inv = torch.argsort(perm)
        full, _ = isab(x)
        shuf, _ = isab(x[:, perm])
        torch.testing.assert_close(full, shuf[:, inv], rtol=1e-5, atol=1e-6)

    def test_isab_bypass_preserves_variable_equivariance_and_gradients(self):
        isab = ISABEncoderLayer(16, 4, 3, time_tokens=4, bypass=True).eval()
        x = torch.randn(2, 11, 16, requires_grad=True)
        perm = torch.cat((torch.randperm(7), torch.arange(7, 11)))
        inv = torch.argsort(perm)
        base, _ = isab(x)
        alternate, _ = isab(x[:, perm])
        torch.testing.assert_close(base, alternate[:, inv], rtol=1e-5, atol=1e-6)
        base.square().mean().backward()
        self.assertIsNotNone(isab.inducing_points.grad)
        self.assertGreater(x.grad[:, -4:].abs().sum().item(), 0)

    def test_luna_contextual_pack_propagation(self):
        luna = LunaEncoder(16, 4, 32, num_layers=3, rank=3, dropout=0).eval()
        self.assertEqual(len(luna.layers), 3)
        x = torch.randn(2, 9, 16, requires_grad=True)
        y, a = luna(x)
        self.assertEqual(tuple(y.shape), (2, 9, 16))
        self.assertEqual(len(a), 3)
        y.square().mean().backward()
        self.assertIsNotNone(luna.initial_p.grad)
        self.assertGreater(luna.initial_p.grad.abs().sum().item(), 0)
        self.assertGreater(luna.layers[0].norm_p.weight.grad.abs().sum().item(), 0)

    def test_luna_equivariance(self):
        luna = LunaEncoder(16, 4, 32, num_layers=2, rank=3, dropout=0).eval()
        x = torch.randn(2, 9, 16)
        perm = torch.randperm(9)
        original, _ = luna(x)
        reord, _ = luna(x[:, perm])
        torch.testing.assert_close(original, reord[:, torch.argsort(perm)],
                                   atol=1e-5, rtol=1e-5)

    def test_luna_bypass_equivariance_and_grad(self):
        luna = LunaEncoder(16, 4, 32, num_layers=2, rank=3, dropout=0,
                           time_tokens=4, bypass=True).eval()
        x = torch.randn(2, 10, 16, requires_grad=True)
        perm = torch.cat((torch.randperm(6), torch.arange(6, 10)))
        y, _ = luna(x)
        yp, _ = luna(x[:, perm])
        torch.testing.assert_close(y, yp[:, torch.argsort(perm)],
                                   atol=1e-5, rtol=1e-5)
        y.square().mean().backward()
        self.assertGreater(x.grad[:, -4:].abs().sum().item(), 0)

    def test_all_heads_and_ranks_finite_backward(self):
        for h, rank in ((1, 1), (2, 3), (4, 6)):
            m = ISABEncoderLayer(16, h, rank, layernorm=True)
            x = torch.randn(2, 11, 16, requires_grad=True)
            y, _ = m(x)
            self.assertTrue(torch.isfinite(y).all())
            y.mean().backward()
            self.assertIsNotNone(x.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
