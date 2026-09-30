"""Mechanism-level numerical validation for the ISAB and Luna adaptations.

Run from installed iTransformer repository root:
  python tests/test_reference_parity.py
  python tests/test_reference_parity.py --official-set-transformer C:\\path\\set_transformer

The optional external test imports the authors' actual set_transformer/modules.py
and compares complete ISAB forward AND backward with identical weights.
Luna is checked against an INDEPENDENT literal implementation of its published
noncausal pack/unpack equations (3)-(6) with separate projection matrices,
including contextual P propagation and all LN/FFN calculations. That is NOT
proof of equality to the full authors' fairseq implementation.
"""
import argparse
import importlib.util
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from layers.ILRA_Comparators import (SetTransformerMAB, ISABEncoderLayer,
                                    LunaEncoder, LunaEncoderLayer)


def attn_formula(mha, queries, context):
    """Independent unpacking of PyTorch MHA projections, softmax and output."""
    d = mha.embed_dim
    qw, kw, vw = mha.in_proj_weight.split(d, 0)
    if mha.in_proj_bias is None:
        qb = kb = vb = None
    else:
        qb, kb, vb = mha.in_proj_bias.split(d, 0)
    q = F.linear(queries, qw, qb)
    k = F.linear(context, kw, kb)
    v = F.linear(context, vw, vb)
    b, nq, _ = q.shape
    nk = k.size(1)
    h, head = mha.num_heads, d // mha.num_heads
    q = q.reshape(b, nq, h, head).transpose(1, 2)
    k = k.reshape(b, nk, h, head).transpose(1, 2)
    v = v.reshape(b, nk, h, head).transpose(1, 2)
    a = (q @ k.transpose(-2, -1) / math.sqrt(head)).softmax(dim=-1)
    result = (a @ v).transpose(1, 2).contiguous().reshape(b, nq, d)
    return F.linear(result, mha.out_proj.weight, mha.out_proj.bias)


def luna_paper_equations(layer, x, p):
    """Full-projection, noncausal Luna equations 3-6, at dropout=0."""
    context = x[:, :-layer.time_tokens] if layer.bypass and layer.time_tokens else x
    raw_pack = attn_formula(layer.pack, p, context)
    if layer.bypass and layer.time_tokens:
        direct_time = x[:, -layer.time_tokens:]
        unpack_context = torch.cat((raw_pack, direct_time), dim=1)
    else:
        unpack_context = raw_pack
    raw_unpack = attn_formula(layer.unpack, x, unpack_context)
    xa = F.layer_norm(x + raw_unpack, (x.size(-1),),
                      layer.norm_x1.weight, layer.norm_x1.bias, layer.norm_x1.eps)
    pa = F.layer_norm(p + raw_pack, (p.size(-1),),
                      layer.norm_p.weight, layer.norm_p.bias, layer.norm_p.eps)
    hidden = layer.activation(F.linear(xa, layer.ff1.weight, layer.ff1.bias))
    ff = F.linear(hidden, layer.ff2.weight, layer.ff2.bias)
    xn = F.layer_norm(xa + ff, (xa.size(-1),),
                      layer.norm_x2.weight, layer.norm_x2.bias, layer.norm_x2.eps)
    return xn, pa


class Parity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def test_luna_published_equations_layer_full_forward_backward(self):
        for bypass, time in ((False, 0), (True, 2)):
            with self.subTest(bypass=bypass):
                l = LunaEncoderLayer(16, 4, 32, dropout=0, time_tokens=time,
                                     bypass=bypass).double().eval()
                x0 = torch.randn(2, 9, 16, dtype=torch.float64)
                p0 = torch.randn(2, 3, 16, dtype=torch.float64)
                x1, p1 = x0.clone().requires_grad_(), p0.clone().requires_grad_()
                x2, p2 = x0.clone().requires_grad_(), p0.clone().requires_grad_()
                yh, ph = l(x1, p1)
                yr, pr = luna_paper_equations(l, x2, p2)
                torch.testing.assert_close(yh, yr, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(ph, pr, rtol=1e-10, atol=1e-10)
                target_params = (l.pack.in_proj_weight, l.unpack.out_proj.weight,
                                 l.norm_p.weight, l.ff1.weight)
                gh = torch.autograd.grad((yh * yh).sum() + (ph * ph).sum(),
                                         (x1, p1) + target_params, retain_graph=False)
                gr = torch.autograd.grad((yr * yr).sum() + (pr * pr).sum(),
                                         (x2, p2) + target_params, retain_graph=False)
                for lhs, rhs in zip(gh, gr):
                    torch.testing.assert_close(lhs, rhs, rtol=2e-9, atol=2e-9)

    def test_luna_multilayer_propagation_matches_paper_equations(self):
        m = LunaEncoder(16, 4, 32, num_layers=3, rank=5,
                        dropout=0).double().eval()
        x = torch.randn(2, 11, 16, dtype=torch.float64)
        actual, _ = m(x)
        p = m.initial_p.expand(x.size(0), -1, -1)
        z = x
        for l in m.layers:
            z, p = luna_paper_equations(l, z, p)
        z = m.norm(z)
        torch.testing.assert_close(actual, z, atol=1e-10, rtol=1e-10)

    def test_isab_scale_matches_official_dim_v_not_head_dim(self):
        m = SetTransformerMAB(16, 16, 16, 4).eval()
        q, k = torch.randn(2, 5, 16), torch.randn(2, 8, 16)
        target = m(q, k)
        projected_q, projected_k, projected_v = m.fc_q(q), m.fc_k(k), m.fc_v(k)
        qa = projected_q.reshape(2, 5, 4, 4).transpose(1, 2)
        ka = projected_k.reshape(2, 8, 4, 4).transpose(1, 2)
        va = projected_v.reshape(2, 8, 4, 4).transpose(1, 2)
        correct_a = ((qa @ ka.transpose(-1, -2)) / math.sqrt(16)).softmax(-1)
        correct_o = (qa + correct_a @ va).transpose(1, 2).reshape(2, 5, 16)
        correct = correct_o + F.relu(m.fc_o(correct_o))
        torch.testing.assert_close(target, correct, atol=1e-6, rtol=1e-6)
        wrong_a = ((qa @ ka.transpose(-1, -2)) / math.sqrt(4)).softmax(-1)
        wrong_o = (qa + wrong_a @ va).transpose(1, 2).reshape(2, 5, 16)
        wrong = wrong_o + F.relu(m.fc_o(wrong_o))
        self.assertGreater((target - wrong).abs().max().item(), 1e-5)


def official_isab_comparison(official_dir):
    """External numerical oracle: actual authors' unmodified ISAB modules.py."""
    file = Path(official_dir) / 'modules.py'
    if not file.exists():
        raise FileNotFoundError(f'Official modules.py not found: {file}')
    spec = importlib.util.spec_from_file_location('original_set_transformer_modules', str(file))
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    for ln in (False, True):
        torch.manual_seed(19)
        src = official.ISAB(16, 16, 4, 3, ln=ln).double().eval()
        ours = ISABEncoderLayer(16, 4, 3, layernorm=ln).double().eval()
        with torch.no_grad():
            ours.inducing_points.copy_(src.I)
        ours.mab0.load_state_dict(src.mab0.state_dict())
        ours.mab1.load_state_dict(src.mab1.state_dict())
        x1 = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        y1 = src(x1)
        y2, _ = ours(x2)
        torch.testing.assert_close(y1, y2, rtol=1e-11, atol=1e-11)
        grad_src = torch.autograd.grad((y1*y1).sum(),
                        (x1, src.I, src.mab0.fc_q.weight, src.mab1.fc_o.weight))
        grad_ours = torch.autograd.grad((y2*y2).sum(),
                        (x2, ours.inducing_points, ours.mab0.fc_q.weight,
                         ours.mab1.fc_o.weight))
        for a, b in zip(grad_src, grad_ours):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        print(f'PASS actual official authors ISAB end-to-end forward/backward, layernorm={ln}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--official-set-transformer', type=str,
                   help='path to cloned authors official set_transformer directory')
    options, remaining = p.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Parity)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if options.official_set_transformer:
        official_isab_comparison(options.official_set_transformer)
    else:
        print('Actual authors ISAB import parity NOT executed: supply --official-set-transformer')
    print('Luna: paper-equation parity only, official fairseq code parity NOT established.')
