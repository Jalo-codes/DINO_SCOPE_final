"""lab_utils.eval.decode_cli — shared argparse wiring for the patch decode.

Every eval suite exposes the SAME flags via ``add_decode_args`` and builds a
``DecodeSpec`` via ``decode_spec_from_args``. Default is ``--decode kmeans`` so
behavior is unchanged unless a caller opts into the calibrated graph decode.

This module is numpy/argparse only (no torch) so it imports cheaply from any
script.
"""

import argparse
from typing import Optional

from lab_utils.eval.partition import DecodeSpec


def add_decode_args(parser: argparse.ArgumentParser) -> None:
    """Register the decode flags on an existing parser.

    Flags:
        --decode {kmeans,graph}     decode method (default kmeans)
        --tau_pos / --tau_neg       trained margins → graph thresholds
        --graph_s_edge              absolute edge similarity (default mid-band)
        --graph_knn                 mutual-kNN k (default 10)
        --graph_spatial             Chebyshev grid radius for spatial-gated edges
        --graph_m_min               minimum component size
        --graph_theta_w/--graph_theta_x  accept thresholds (default derived)
        --graph_attention_polarity  background = lowest-attention large component
    """
    g = parser.add_argument_group('decode')
    g.add_argument('--decode', choices=('kmeans', 'graph'), default='kmeans',
                   help='Patch decode method (default: kmeans). "graph" uses the '
                        'calibrated connected-components decode.')
    g.add_argument('--tau_pos', type=float, default=0.55,
                   help='Same-region cohesion floor (graph decode default '
                        'thresholds derive from this; match the trained value).')
    g.add_argument('--tau_neg', type=float, default=0.20,
                   help='Cross-region separation ceiling (graph decode).')
    g.add_argument('--graph_s_edge', type=float, default=None,
                   help='Absolute edge similarity bar. Default (tau_pos+tau_neg)/2.')
    g.add_argument('--graph_knn', type=int, default=10,
                   help='Mutual-kNN k for graph edges (anti-chaining).')
    g.add_argument('--graph_spatial', type=int, default=None,
                   help='If set, edges also require Chebyshev grid distance ≤ r.')
    g.add_argument('--graph_m_min', type=int, default=4,
                   help='Ignore components smaller than this many patches.')
    g.add_argument('--graph_theta_w', type=float, default=None,
                   help='Accept iff internal cohesion ≥ this. Default tau_pos-0.05.')
    g.add_argument('--graph_theta_x', type=float, default=None,
                   help='Accept iff mean similarity to background ≤ this. '
                        'Default (tau_pos+tau_neg)/2.')
    g.add_argument('--graph_attention_polarity', action='store_true', default=False,
                   help='Pick background among large components by lowest mean '
                        'attention (handles >50%% splices). Default off.')


def decode_spec_from_args(args, *, n_init: int = 4) -> DecodeSpec:
    """Build a ``DecodeSpec`` from a parsed args namespace produced by
    ``add_decode_args``. Tolerates missing attributes (falls back to defaults)."""
    def _get(name, default):
        return getattr(args, name, default)
    return DecodeSpec(
        method=_get('decode', 'kmeans'),
        tau_pos=float(_get('tau_pos', 0.55)),
        tau_neg=float(_get('tau_neg', 0.20)),
        n_init=int(n_init),
        s_edge=_get('graph_s_edge', None),
        mutual_knn_k=int(_get('graph_knn', 10)),
        r_spatial=_get('graph_spatial', None),
        m_min=int(_get('graph_m_min', 4)),
        theta_w=_get('graph_theta_w', None),
        theta_x=_get('graph_theta_x', None),
        attention_polarity=bool(_get('graph_attention_polarity', False)),
    )


def decode_label(spec: DecodeSpec) -> str:
    """Short human-readable tag for logs/report headers."""
    if spec.method == 'kmeans':
        return 'kmeans'
    bits = [f's={spec.s_edge if spec.s_edge is not None else "mid"}',
            f'k={spec.mutual_knn_k}']
    if spec.r_spatial is not None:
        bits.append(f'r={spec.r_spatial}')
    return 'graph[' + ','.join(bits) + ']'
