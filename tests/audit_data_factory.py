"""Audit that evaluation uses original iTransformer data_provider behavior.

The actual input dataset and splits are not checked here. This structural
check avoids accidentally using an earlier research patch that rewired val/test.
"""
import argparse
import ast
import sys
from pathlib import Path

EXPECTED = '''
def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
'''


def function(tree_text, filename):
    module = ast.parse(tree_text, filename=str(filename))
    matches = [n for n in module.body if isinstance(n, ast.FunctionDef)
               and n.name == 'data_provider']
    if len(matches) != 1:
        raise RuntimeError(f'Expected data_provider exactly once in {filename}')
    return matches[0]


def simplified(node):
    """Keep AST semantics, normalize missing docstrings/comment-free formatting."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()
    here = Path(__file__).resolve().parents[1]
    p = here / 'data_provider' / 'data_factory.py'
    if not p.exists():
        print('UNKNOWN: missing data_factory.py (must run in real iTransformer repo)')
        if args.strict:sys.exit(2)
        return
    expected = simplified(function(EXPECTED, 'upstream reference'))
    actual = simplified(function(p.read_text(encoding='utf-8-sig'), p))
    if expected != actual:
        print('MISMATCH: data_provider differs structurally from public original iTransformer')
        print('Review upstream https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py')
        print('Do not mix differing validation/test DataLoader behavior in comparisons.')
        if args.strict:sys.exit(2)
    else:
        print('PASS: data_provider function AST equivalent to original iTransformer')


if __name__=='__main__':
    main()
