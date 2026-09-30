"""Verify local ILRA operator AST equals the EXACT operator in user's newest patch.

This checks the central ILRA operator (including read/write, bypass and gate).
It does NOT assert equality of the full repo, training logs or checkpoints.
Run from an iTransformer repository after installing comparison kit v2:
    python tests/verify_latest_ilra_source.py
"""
import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'tests' / 'ILRA_operator_reference.py.txt'
TARGET = ROOT / 'layers' / 'SelfAttention_Family.py'


def cls_ast(text, filename):
    mod = ast.parse(text, filename=str(filename))
    candidates = [x for x in mod.body
                  if isinstance(x, ast.ClassDef) and x.name == 'InducedVariateAttention']
    if len(candidates) != 1:
        raise RuntimeError(f'{filename}: expected 1 InducedVariateAttention class, got {len(candidates)}')
    return ast.dump(candidates[0], annotate_fields=True, include_attributes=False)


def main():
    if not TARGET.exists() or not REF.exists():
        raise RuntimeError('Run this after installation from the iTransformer root')
    ref_ast = cls_ast(REF.read_text(encoding='utf-8'), REF)
    live_ast = cls_ast(TARGET.read_text(encoding='utf-8'), TARGET)
    ref_hash = hashlib.sha256(ref_ast.encode()).hexdigest()
    got_hash = hashlib.sha256(live_ast.encode()).hexdigest()
    print('Latest user patch class AST SHA256 :', ref_hash)
    print('Installed ILRA class AST SHA256  :', got_hash)
    if got_hash != ref_hash:
        raise AssertionError('ILRA central operator IS NOT identical to the supplied latest patch')
    print('PASS: exact source-level AST equality for central ILRA operator')


if __name__ == '__main__':
    main()
