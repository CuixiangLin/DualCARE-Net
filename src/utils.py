# -*- coding: utf-8 -*-
"""Utilities: evaluation metrics, hypergraph G construction, label and index handling."""
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve


# ----------------- Evaluation -----------------
def cal_auc(output, labels):
    """output: log_softmax output (N, 2); labels: (N,)"""
    out = output.cpu().detach().numpy()
    out = np.exp(out)[:, 1]
    y   = labels.cpu().numpy()
    auroc = roc_auc_score(y, out)
    p, r, _ = precision_recall_curve(y, out)
    auprc = auc(r, p)
    return auroc, auprc


# ----------------- Hypergraph G (weighted Laplacian) -----------------
def generate_G_from_H_weight_gpu(H_np, W_np, device):
    """G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}, computed on the GPU.

    A direct implementation with explicit diagonal matrices ends in an N^3
    matrix product. This version rescales element-wise and performs a single
    (N, M) @ (M, N) product, which is mathematically equivalent.

    H_np : (N, M) float ndarray - zero-degree nodes are fixed by the caller
    W_np : (M,)  float ndarray - hyperedge weights
    Returns : (N, N) float32 tensor on device
    """
    H  = torch.from_numpy(H_np).float().to(device)          # (N, M)
    W  = torch.from_numpy(W_np).float().to(device)          # (M,)
    DE = H.sum(dim=0).clamp(min=1e-12)                      # (M,) hyperedge degrees
    DV = (H * W).sum(dim=1).clamp(min=1e-12)                # (N,) weighted node degrees
    H_scaled = H * (W / DE)                                  # column scaling: W[j]/DE[j]
    G  = H_scaled @ H.t()                                   # (N, N)
    dv = DV.pow(-0.5)
    G  = dv.unsqueeze(1) * G * dv.unsqueeze(0)              # row and column scaling
    return G


def compute_fold_G(train_idx, label_frame, incidence_matrix,
                   device, hyperedge_min_positive):
    """Build the disease-specific hypergraph G from the positives of the training fold.

    Hyperedge selection and weighting use training-fold labels only; test labels
    are never accessed here.
    """
    train_frame = label_frame.iloc[train_idx]
    train_pos   = list(train_frame.where(train_frame == 1).dropna().index)
    pos_sum     = incidence_matrix.loc[train_pos].sum()

    sel_idx  = np.where(pos_sum >= hyperedge_min_positive)[0]
    sel_H    = incidence_matrix.iloc[:, sel_idx]
    he_w     = pos_sum[sel_idx].values
    he_total = incidence_matrix.iloc[:, sel_idx].values.sum(0)
    he_w     = he_w / he_total                               # hyperedge weight = share of positives

    H  = np.array(sel_H).astype('float')
    DV = np.sum(H * he_w, axis=1)
    for i in range(H.shape[0]):                              # guard against zero-degree nodes
        if DV[i] == 0:
            H[i][random.randint(0, H.shape[1] - 1)] = 1e-4

    return generate_G_from_H_weight_gpu(H, he_w, device)


# ----------------- Incidence matrix: MSigDB c2/c5 -----------------
def geneset_keep_mask(data_dir):
    """
    Which gene sets survive the DISHyper / DISFusion disease-keyword filter.
    Returns {'c2': [...], 'c5': [...]}, one flag per gene set in the order
    c2Name.txt / c5Name.txt list them:
      - all c2 entries: dropped when a name token is CANCER / TUMOR / NEOPLASM
      - c5 entries starting with 'HP': same rule
      - remaining c5 entries (GO-derived): kept without disease-keyword filtering
    The hypergraph (H) and the gene-set feature matrix (B) are both filtered with
    this, so the two encoders operate on the same gene sets.
    """
    import os
    BAD = {'CANCER', 'TUMOR', 'NEOPLASM'}
    mask = {}
    for gid in ['c2', 'c5']:
        names = pd.read_csv(os.path.join(data_dir, f'{gid}Name.txt'),
                            sep='\t', header=None)
        keep = []
        for name in names[0].values:
            if gid == 'c2' or name[:2] == 'HP':
                keep.append(not (BAD & set(name.split('_'))))
            else:
                keep.append(True)
        mask[gid] = keep
    return mask


def processing_incidence_matrix(gene_list, data_dir):
    """
    Follows the inputs and filtering rules of DISHyper / DISFusion:
      c2/c5_GenesetsMatrix.npz : gene x gene-set 0/1 incidence matrix
      c2/c5Name.txt            : gene-set names
    The keyword filter itself lives in geneset_keep_mask.
    """
    import os
    mask = geneset_keep_mask(data_dir)
    incidence = pd.DataFrame(index=gene_list)
    for gid in ['c2', 'c5']:
        keep = [z for z, k in enumerate(mask[gid]) if k]
        mat = sp.load_npz(os.path.join(data_dir, f'{gid}_GenesetsMatrix.npz'))
        df  = pd.DataFrame(data=mat.toarray(), index=gene_list).iloc[:, keep]
        incidence = pd.concat([incidence, df], axis=1)
    incidence.columns = np.arange(incidence.shape[1])
    return incidence


# ----------------- Positives, negatives and labels -----------------
def get_data(positive_path, negative_path, gene_list):
    pos = pd.read_csv(positive_path, header=None)[0].tolist()
    pos = sorted(set(gene_list) & set(pos))
    neg = pd.read_csv(negative_path, header=None)[0].tolist()
    neg = sorted(set(gene_list) & set(neg))

    label_frame = pd.DataFrame(data=[0] * len(gene_list), index=gene_list)
    label_frame.loc[pos, :] = 1
    pos_idx = list(np.where(label_frame == 1)[0])

    label_frame.loc[neg, :] = -1
    neg_idx = list(np.where(label_frame == -1)[0])

    label_frame = pd.DataFrame(data=[0] * len(gene_list), index=gene_list)
    label_frame.loc[pos, :] = 1

    sample_idx = np.array(pos_idx + neg_idx)
    label = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
    return sample_idx, label, label_frame
