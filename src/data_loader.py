# -*- coding: utf-8 -*-
"""
Data loading for DualCARE-Net.

  load_gene_list         : the 13,627 gene symbols that define the global node order
  load_node_features     : 98-dim molecular features, aligned to gene_list
  load_weighted_ppi      : weighted PPI -> edge_index (2, 2E), edge_weight (2E,)
                           undirected graph => both directions plus self-loops
  load_hypergraph        : gene x gene-set incidence matrix
  load_geneset_features  : binary c2/c5 gene-set membership features (B encoder)
  load_ppi_dense_matrix  : dense PPI adjacency matrix (S encoder)
"""
import numpy as np
import pandas as pd
import torch

from . import config as cfg


def load_gene_list(path=None):
    path = path or cfg.GENE_LIST_PATH
    genes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 2:
                genes.append(p[1])
    return genes


def load_node_features(gene_list, path=None):
    """Read the 98-dim features, align them to gene_list, fill missing with 0."""
    path = path or cfg.MULTIOMICS_FEATURE_PATH
    df = pd.read_csv(path, index_col='Gene')
    df = df.reindex(gene_list, fill_value=0.0)
    arr = df.values.astype(np.float32)
    print(f'[features] shape={arr.shape}, '
          f'mean={arr.mean():.4f}, std={arr.std():.4f}')
    return torch.from_numpy(arr)


def load_weighted_ppi(gene_list,
                      version=None,
                      alpha=None,
                      add_self_loops=True):
    """
    Returns:
      edge_index : LongTensor (2, 2E + N)
      edge_weight: FloatTensor (2E + N,)

    Edge weights follow the unified formulation of the paper:
        w_ij = (1 - alpha) * conf_ij + alpha * mutcos(i, j)
    With alpha = 0 this reduces to the confidence-weighted PPI, the pan-cancer
    setting reported in the paper. Any alpha above 0 requires the PPI file to
    carry a 'mutcos' column.
    """
    version = version or cfg.PPI_VERSION
    alpha   = cfg.ALPHA if alpha is None else alpha
    path    = cfg.PPI_PATHS[version]

    df = pd.read_csv(path, sep='\t')
    gene2idx = {g: i for i, g in enumerate(gene_list)}

    # Drop the few edges with an endpoint outside gene_list
    mask = df['g1'].isin(gene2idx) & df['g2'].isin(gene2idx)
    df = df[mask].reset_index(drop=True)

    src = df['g1'].map(gene2idx).values
    dst = df['g2'].map(gene2idx).values

    if alpha > 0:
        if 'mutcos' not in df.columns:
            raise ValueError(
                f'alpha={alpha} requires a "mutcos" column in {path}, but that file '
                'provides confidence scores only. Use alpha=0 to reproduce the '
                'pan-cancer setting of the paper.'
            )
        w = ((1 - alpha) * df['conf'].values
             + alpha * df['mutcos'].values).astype(np.float32)
        weight_kind = f'(1-{alpha})*conf + {alpha}*mutcos'
    else:
        w = df['conf'].values.astype(np.float32)
        weight_kind = 'conf only (alpha=0)'

    # Undirected -> both directions
    edge_index = np.concatenate(
        [np.stack([src, dst]), np.stack([dst, src])], axis=1
    )
    edge_weight = np.concatenate([w, w])

    # Self-loops with weight 1.0, so isolated genes still carry information
    if add_self_loops:
        N = len(gene_list)
        loop = np.stack([np.arange(N), np.arange(N)])
        loop_w = np.ones(N, dtype=np.float32)
        edge_index = np.concatenate([edge_index, loop], axis=1)
        edge_weight = np.concatenate([edge_weight, loop_w])

    print(f'[ppi-{version}] edges={len(df):,}, '
          f'directed_arcs+selfloops={edge_index.shape[1]:,}, '
          f'weight: {weight_kind}, '
          f'mean={w.mean():.3f}, min={w.min():.3f}, max={w.max():.3f}')

    return (torch.from_numpy(edge_index).long(),
            torch.from_numpy(edge_weight).float())


def load_ppi_dense_matrix(gene_list, edge_index, edge_weight):
    """Build the dense symmetric N x N PPI adjacency matrix for the S encoder.

    At N = 13,627 this occupies about 708 MiB of GPU memory in float32.
    """
    N = len(gene_list)
    A = torch.zeros(N, N, dtype=torch.float32, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = edge_weight
    nz = (A != 0).float().mean().item()
    print(f'[ppi-dense] shape=({N},{N}), nonzero_ratio={nz:.4f}, '
          f'mean={A.mean():.4f}, max={A.max():.4f}')
    return A


def load_geneset_features(gene_list, path=None, data_dir=None):
    """Read the binary c2/c5 gene-set membership matrix (gene x gene-set) for the
    B encoder, aligned to gene_list with 0 for missing genes.

    The file on disk holds the full c2+c5 pool. The disease-keyword filter that the
    hypergraph uses is applied here as well, so the H and B encoders operate on the
    same gene sets.
    """
    from .utils import geneset_keep_mask
    path = path or cfg.GENESET_FEATURE_PATH
    data_dir = data_dir or cfg.DATA_DIR
    df = pd.read_csv(path, sep='\t', index_col=0)

    mask = geneset_keep_mask(data_dir)
    flat = mask['c2'] + mask['c5']
    if df.shape[1] != len(flat):
        raise ValueError(f'gene-set feature matrix has {df.shape[1]} columns, '
                         f'expected the c2+c5 pool of {len(flat)}')
    df = df.iloc[:, [i for i, k in enumerate(flat) if k]]

    df = df.reindex(gene_list, fill_value=0.0)
    arr = df.values.astype(np.float32)
    nz = (arr != 0).mean()
    print(f'[geneset-feat] shape={arr.shape} (pool {len(flat)}, kept {sum(flat)}), '
          f'nonzero_ratio={nz:.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}')
    return torch.from_numpy(arr)


def load_hypergraph(gene_list, data_dir=None):
    """Gene x gene-set incidence matrix; the disease-specific weighting is applied
    per fold inside the training loop.
    """
    from .utils import processing_incidence_matrix
    data_dir = data_dir or cfg.DATA_DIR
    inc = processing_incidence_matrix(gene_list, data_dir)
    print(f'[hypergraph] incidence={inc.shape}')
    return inc
