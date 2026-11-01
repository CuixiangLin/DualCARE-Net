# -*- coding: utf-8 -*-
"""
DualCARE-Net entry point. The default configuration is the full HCMBS model:
  python main.py
"""
import os
import argparse
import random
import numpy as np
import torch

from src import config as cfg
from src.data_loader import (
    load_gene_list, load_node_features,
    load_weighted_ppi, load_hypergraph,
    load_geneset_features, load_ppi_dense_matrix,
)
from src.train import train_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('output', nargs='?', default='results/pred_HCMBS_best.tsv',
                   help='output .tsv path (default: results/pred_HCMBS_best.tsv)')
    p.add_argument('--ppi',   default=cfg.PPI_VERSION,
                   choices=list(cfg.PPI_PATHS.keys()))
    p.add_argument('--alpha', type=float, default=cfg.ALPHA)
    p.add_argument('--feature', default=cfg.MULTIOMICS_FEATURE_PATH)
    p.add_argument('--epochs',        type=int,   default=cfg.EPOCHS)
    p.add_argument('--seed',          type=int,   default=cfg.SEED)
    p.add_argument('--n_repeats',     type=int,   default=cfg.N_REPEATS)
    # ---------- Shared structure ----------
    p.add_argument('--n_hid',         type=int,   default=cfg.N_HID)
    p.add_argument('--dropout',       type=float, default=cfg.DROPOUT)
    # ---------- Per-encoder learning rates ----------
    p.add_argument('--lr_h',          type=float, default=cfg.LR_H)
    p.add_argument('--wd_h',          type=float, default=cfg.WD_H)
    p.add_argument('--lr_c',          type=float, default=cfg.LR_C)
    p.add_argument('--wd_c',          type=float, default=cfg.WD_C)
    p.add_argument('--lr_m',          type=float, default=cfg.LR_M)
    p.add_argument('--wd_m',          type=float, default=cfg.WD_M)
    p.add_argument('--lr_b',          type=float, default=cfg.LR_B)
    p.add_argument('--wd_b',          type=float, default=cfg.WD_B)
    p.add_argument('--lr_s',          type=float, default=cfg.LR_S)
    p.add_argument('--wd_s',          type=float, default=cfg.WD_S)
    # ---------- Fusion head ----------
    p.add_argument('--lr_fusion',     type=float, default=cfg.LR_FUSION)
    p.add_argument('--wd_fusion',     type=float, default=cfg.WD_FUSION)
    p.add_argument('--lambda_inter',  type=float, default=cfg.LAMBDA_INTER)
    p.add_argument('--w_self',        type=float, default=cfg.W_SELF)
    # ---------- Encoder selection ----------
    p.add_argument('--encoders', default=cfg.ENCODERS,
                   help='comma-separated subset of {H,C,M,B,S}, at least 1')
    # ---------- Fusion mode ----------
    p.add_argument('--attention', action='store_true', default=False,
                   help='use additive attention fusion (default: concatenation)')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg.PPI_VERSION   = args.ppi
    cfg.ALPHA         = args.alpha
    cfg.EPOCHS        = args.epochs
    cfg.N_HID         = args.n_hid
    cfg.DROPOUT       = args.dropout
    cfg.LR_H          = args.lr_h
    cfg.WD_H          = args.wd_h
    cfg.LR_C          = args.lr_c
    cfg.WD_C          = args.wd_c
    cfg.LR_M          = args.lr_m
    cfg.WD_M          = args.wd_m
    cfg.LR_B          = args.lr_b
    cfg.WD_B          = args.wd_b
    cfg.LR_S          = args.lr_s
    cfg.WD_S          = args.wd_s
    cfg.LR_FUSION     = args.lr_fusion
    cfg.WD_FUSION     = args.wd_fusion
    cfg.LAMBDA_INTER  = args.lambda_inter
    cfg.W_SELF        = args.w_self
    cfg.N_REPEATS     = args.n_repeats
    cfg.ENCODERS      = args.encoders
    cfg.USE_ATTENTION = args.attention
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    print(f'[setup ] ppi={args.ppi}  alpha={args.alpha}  '
          f'feat_dim={cfg.FEATURE_DIM}  epochs={args.epochs}  '
          f'n_hid={args.n_hid}  dropout={args.dropout}')
    print(f'[lr    ] H={args.lr_h}  C={args.lr_c}  '
          f'M={args.lr_m}  B={args.lr_b}  S={args.lr_s}')
    print(f'[fusion] lr_fusion={args.lr_fusion}  wd_fusion={args.wd_fusion}  '
          f'lambda_inter={args.lambda_inter}  w_self={args.w_self}')
    print(f'[encoders] {args.encoders}')

    # ---------- Data ----------
    gene_list = load_gene_list()
    print(f'[gene  ] N={len(gene_list):,}')

    node_feat = load_node_features(gene_list, args.feature).to(device)
    edge_index, edge_weight = load_weighted_ppi(
        gene_list, version=args.ppi, alpha=args.alpha
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    incidence  = load_hypergraph(gene_list)

    # Load the gene-set features only when the B encoder is selected
    # (they occupy more than 1 GB of GPU memory)
    encoder_set = {s.strip().upper() for s in args.encoders.split(',')}
    geneset_feat = None
    if 'B' in encoder_set:
        geneset_feat = load_geneset_features(gene_list).to(device)

    # Build the dense PPI matrix only when the S encoder is selected
    # (13,627 x 13,627 float32 is about 708 MiB of GPU memory)
    adj_matrix = None
    if 'S' in encoder_set:
        adj_matrix = load_ppi_dense_matrix(gene_list, edge_index, edge_weight)

    # ---------- Training ----------
    auroc_list, auprc_list, eval_res = train_pred(
        gene_list, node_feat, edge_index, edge_weight, incidence,
        cfg.POSITIVE_PATH, cfg.NEGATIVE_PATH, device,
        geneset_features=geneset_feat,
        adj_matrix=adj_matrix,
    )

    # ---------- Output ----------
    n_runs = cfg.N_REPEATS * cfg.N_SPLITS
    pred = eval_res.sum(1).sort_values(ascending=False) / n_runs
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    pred.to_csv(args.output, sep='\t', header=False)

    print('=' * 55)
    print(f'AUROC  mean={np.mean(auroc_list):.4f}  std={np.std(auroc_list):.4f}')
    print(f'AUPRC  mean={np.mean(auprc_list):.4f}  std={np.std(auprc_list):.4f}')
    print(f'pred -> {args.output}')


if __name__ == '__main__':
    main()
