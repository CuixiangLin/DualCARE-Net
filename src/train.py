# -*- coding: utf-8 -*-
"""
Training and evaluation loop for DualCARE-Net: one optimizer per encoder plus
one for the fusion head, evaluated by repeated stratified 5-fold cross-validation.

Five encoders (--encoders H,C,M,B,S; any subset of 1-5 can be selected):
  H : DISHyperHGNN     hypergraph encoder      (input fh = N x N identity, disease-specific G)
  C : WeightedChebNet  Chebyshev graph encoder (98-dim features over the weighted PPI)
  M : MultiOmicsLinear molecular encoder       (98-dim features, no propagation)
  B : GeneSetLinear    gene-set encoder        (binary c2/c5 features, no propagation)
  S : AdjacencyLinear  PPI adjacency encoder   (row A[i, :] of the dense adjacency, no propagation)

FusionHead concatenates (or attention-weights) the selected encoder embeddings,
and the Barlow Twins redundancy-reduction loss is summed over all encoder pairs.
"""
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .models import (
    WeightedChebNet, DISHyperHGNN, MultiOmicsLinear, GeneSetLinear,
    AdjacencyLinear,
    FusionHead,
)
from .utils import cal_auc, compute_fold_G, get_data

VALID_ENCODERS = ('H', 'C', 'M', 'B', 'S')


def _parse_encoders(encoders_str):
    """Parse 'H,C,M' -> ['H', 'C', 'M'] and validate."""
    keys = [s.strip().upper() for s in encoders_str.split(',') if s.strip()]
    for k in keys:
        if k not in VALID_ENCODERS:
            raise ValueError(f'unknown encoder {k}, must be one of {VALID_ENCODERS}')
    if len(keys) < 1:
        raise ValueError(f'at least one encoder is required, got: {keys}')
    if len(set(keys)) != len(keys):
        raise ValueError(f'encoders must not repeat: {keys}')
    return keys


def _build_encoder(key, N, geneset_dim, device):
    """Build the encoder for the given key; returns (module, lr, weight_decay)."""
    if key == 'H':
        m = DISHyperHGNN(in_ch=N, n_hid=cfg.N_HID,
                         num_layers=cfg.N_LAYERS, dropout=0.2)
        return m.to(device), cfg.LR_H, cfg.WD_H
    if key == 'C':
        m = WeightedChebNet(in_dim=cfg.FEATURE_DIM, hdim=cfg.N_HID,
                            K=cfg.CHEB_K, dropout=cfg.DROPOUT)
        return m.to(device), cfg.LR_C, cfg.WD_C
    if key == 'M':
        m = MultiOmicsLinear(in_dim=cfg.FEATURE_DIM, hdim=cfg.N_HID,
                             dropout=cfg.DROPOUT)
        return m.to(device), cfg.LR_M, cfg.WD_M
    if key == 'B':
        assert geneset_dim is not None, 'the B encoder requires geneset_features'
        m = GeneSetLinear(in_dim=geneset_dim, hdim=cfg.N_HID,
                          dropout=cfg.DROPOUT)
        return m.to(device), cfg.LR_B, cfg.WD_B
    if key == 'S':
        m = AdjacencyLinear(in_dim=N, hdim=cfg.N_HID, dropout=cfg.DROPOUT)
        return m.to(device), cfg.LR_S, cfg.WD_S
    raise ValueError(key)


def _forward_encoder(key, model, fh, G_hyper, node_features,
                     edge_index, edge_weight, geneset_features, adj_matrix):
    """Unified forward pass for a single encoder."""
    if key == 'H': return model(fh, G_hyper)
    if key == 'C': return model(node_features, edge_index, edge_weight)
    if key == 'M': return model(node_features)
    if key == 'B': return model(geneset_features)
    if key == 'S': return model(adj_matrix)
    raise ValueError(key)


# =================================================================
# One fold: train and test
# =================================================================
def _train_one_fold(
    train_idx, test_idx,
    label_frame,
    node_features, edge_index, edge_weight, geneset_features, adj_matrix,
    gene_list, device,
    G_hyper, fh, encoder_keys,
):
    N           = len(gene_list)
    geneset_dim = geneset_features.size(1) if geneset_features is not None else None
    labels      = torch.from_numpy(label_frame.values.reshape(-1)).long().to(device)

    # ---------- Build encoders and their optimizers ----------
    encoders  = {}    # key -> module
    opt_dict  = {}    # key -> optimizer
    for k in encoder_keys:
        m, lr, wd = _build_encoder(k, N, geneset_dim, device)
        encoders[k] = m
        opt_dict[k] = optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    fusion = FusionHead(
        input_dim=cfg.N_HID, n_encoders=len(encoder_keys),
        lambd_inter=cfg.LAMBDA_INTER,
        attention=cfg.USE_ATTENTION, n_class=cfg.N_CLASS, dropout=cfg.DROPOUT,
    ).to(device)
    opt_fuse = optim.Adam(fusion.parameters(),
                          lr=cfg.LR_FUSION, weight_decay=cfg.WD_FUSION)

    # The LR scheduler is attached only when the H encoder is present
    sched_hyper = None
    if 'H' in encoders:
        sched_hyper = optim.lr_scheduler.MultiStepLR(
            opt_dict['H'], milestones=cfg.LR_MILESTONES, gamma=cfg.LR_GAMMA
        )

    # ---------- Training loop ----------
    for epoch in range(cfg.EPOCHS):
        for m in encoders.values(): m.train()
        fusion.train()
        for o in opt_dict.values(): o.zero_grad()
        opt_fuse.zero_grad()

        h_list = [
            _forward_encoder(k, encoders[k], fh, G_hyper,
                             node_features, edge_index, edge_weight,
                             geneset_features, adj_matrix)
            for k in encoder_keys
        ]
        loss_inter, semi = fusion(h_list)
        loss = F.nll_loss(semi[train_idx], labels[train_idx]) \
               + cfg.W_SELF * loss_inter
        loss.backward()

        opt_fuse.step()
        for o in opt_dict.values(): o.step()
        if sched_hyper is not None: sched_hyper.step()

    # ---------- Test ----------
    for m in encoders.values(): m.eval()
    fusion.eval()
    with torch.no_grad():
        h_list = [
            _forward_encoder(k, encoders[k], fh, G_hyper,
                             node_features, edge_index, edge_weight,
                             geneset_features, adj_matrix)
            for k in encoder_keys
        ]
        _, semi = fusion(h_list)
        auroc, auprc = cal_auc(semi[test_idx], labels[test_idx])
        prob = pd.DataFrame(
            data=semi.exp().cpu().numpy(), index=gene_list
        )
    return auroc, auprc, prob


# =================================================================
# Repeated stratified 5-fold cross-validation
# =================================================================
def train_pred(
    gene_list, node_features,
    edge_index, edge_weight,
    incidence_matrix,
    positive_path, negative_path,
    device,
    geneset_features=None,
    adj_matrix=None,
):
    encoder_keys = _parse_encoders(cfg.ENCODERS)
    print(f'[encoders] active={encoder_keys}  (n={len(encoder_keys)})')

    if 'B' in encoder_keys and geneset_features is None:
        raise ValueError('the B encoder was selected but geneset_features was not provided')
    if 'S' in encoder_keys and adj_matrix is None:
        raise ValueError('the S encoder was selected but adj_matrix was not provided')

    auroc_list, auprc_list = [], []
    eval_res = pd.DataFrame(index=gene_list)

    N  = len(gene_list)
    fh = torch.eye(N, device=device) if 'H' in encoder_keys else None

    for repeat in range(cfg.N_REPEATS):
        sample_idx, label, label_frame = get_data(
            positive_path, negative_path, gene_list
        )
        sk_X = sample_idx.reshape([-1, 1])
        skf = StratifiedKFold(
            n_splits=cfg.N_SPLITS, random_state=repeat, shuffle=True
        )
        for fold, (tr, te) in enumerate(skf.split(sk_X, label)):
            train_idx = sample_idx[tr]
            test_idx  = sample_idx[te]

            G_hyper = None
            if 'H' in encoder_keys:
                G_hyper = compute_fold_G(
                    train_idx, label_frame, incidence_matrix,
                    device, cfg.HYPEREDGE_MIN_POSITIVE
                )

            auroc, auprc, prob = _train_one_fold(
                train_idx, test_idx, label_frame,
                node_features, edge_index, edge_weight, geneset_features, adj_matrix,
                gene_list, device,
                G_hyper=G_hyper, fh=fh, encoder_keys=encoder_keys,
            )
            auroc_list.append(float(auroc))
            auprc_list.append(float(auprc))
            eval_res = pd.concat([eval_res, prob[1]], axis=1)
            print(f'  [repeat {repeat+1}/{cfg.N_REPEATS} fold {fold+1}/{cfg.N_SPLITS}]'
                  f'  AUROC={auroc:.4f}  AUPRC={auprc:.4f}')

    return auroc_list, auprc_list, eval_res
