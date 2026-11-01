# -*- coding: utf-8 -*-
"""
Encoder modules and the redundancy-reduced fusion head of DualCARE-Net.

Nonlinear branch:
  WeightedChebNet  : C encoder - Chebyshev graph convolution over the weighted PPI
                     (supports continuous edge weights)
  DISHyperHGNN     : H encoder - residual hypergraph convolution
                     (the disease-specific G is built outside this module)

Linear branch (two fully connected layers each, no nonlinear activation):
  MultiOmicsLinear : M encoder - 98-dim molecular features, no propagation
  GeneSetLinear    : B encoder - binary c2/c5 gene-set membership features
  AdjacencyLinear  : S encoder - row A[i, :] of the weighted PPI adjacency matrix

  FusionHead       : concatenates the encoder embeddings and applies a pairwise
                     Barlow Twins objective over all encoder pairs
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import ChebConv


# =================================================================
# C encoder: Chebyshev graph convolution over the weighted PPI
# =================================================================
class WeightedChebNet(nn.Module):
    def __init__(self, in_dim, hdim=256, K=2, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hdim, K=K)
        self.conv2 = ChebConv(hdim,  hdim, K=K)
        self.conv3 = ChebConv(hdim,  hdim, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x


# =================================================================
# H encoder: residual hypergraph convolution
# =================================================================
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return G.matmul(x)


class ResHGNNLayer(nn.Module):
    def __init__(self, n_hid, dropout=0.5):
        super().__init__()
        self.hgc = HGNN_conv(n_hid, n_hid)
        self.dropout = dropout

    def forward(self, x, G):
        h = self.hgc(x, G)
        h = F.relu(h + x)
        return F.dropout(h, p=self.dropout, training=self.training)


class DISHyperHGNN(nn.Module):
    def __init__(self, in_ch, n_hid, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch, n_hid)
        self.layers = nn.ModuleList(
            [ResHGNNLayer(n_hid, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x, G):
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, G)
        return x


# =================================================================
# M encoder: molecular features, no graph propagation
#   Two fully connected layers without a nonlinear activation, so the
#   transformation stays linear.
# =================================================================
class MultiOmicsLinear(nn.Module):
    def __init__(self, in_dim, hdim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# =================================================================
# B encoder: binary c2/c5 gene-set membership features
#   Same gene-set evidence as the H encoder, encoded linearly instead of
#   propagated over the hypergraph.
# =================================================================
class GeneSetLinear(nn.Module):
    def __init__(self, in_dim, hdim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# =================================================================
# S encoder: weighted PPI adjacency vectors
#   Input is row A[i, :] of the dense adjacency matrix, i.e. the weighted
#   adjacency vector of gene i. Same structure as M and B; input dim = N.
# =================================================================
class AdjacencyLinear(nn.Module):
    def __init__(self, in_dim, hdim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# =================================================================
# Fusion head: fuse n encoder embeddings and classify
#   - combine_concat / combine_att
#   - loss_inter : pairwise Barlow Twins, summed over all encoder pairs
# =================================================================
def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LogReg(nn.Module):
    def __init__(self, in_dim, n_class, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_class)
        self.dropout = dropout
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.fc(x), dim=1)


class FusionHead(nn.Module):
    def __init__(self, input_dim, n_encoders, lambd_inter,
                 attention=False, n_class=2, dropout=0.5):
        super().__init__()
        assert n_encoders in (1, 2, 3, 4, 5), 'n_encoders must be 1/2/3/4/5'
        self.n_encoders  = n_encoders
        self.lambd_inter = lambd_inter
        self.attention   = attention
        self.dropout     = dropout

        self.w_list = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_encoders)]
        )
        self.y_list = nn.ModuleList(
            [nn.Linear(input_dim, 1) for _ in range(n_encoders)]
        )
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

        self.concat_fc = nn.Linear(input_dim * n_encoders, input_dim)
        self.logistic  = LogReg(input_dim, n_class, dropout)

    def combine_att(self, h_list):
        scores = [self.y_list[i](self.w_list[i](h)) for i, h in enumerate(h_list)]
        score  = torch.cat(scores, -1)
        score  = self.att_act2(self.att_act1(score)).unsqueeze(-1)
        h_stack = torch.stack(h_list, dim=1)
        return (score * h_stack).sum(dim=1)

    def combine_concat(self, h_list):
        return F.relu(self.concat_fc(torch.cat(h_list, dim=1)))

    @staticmethod
    def _barlow_pair(h1, h2, lambd):
        EPS = 1e-15
        N   = h1.size(0)
        z1  = (h1 - h1.mean(0)) / (h1.std(0) + EPS)
        z2  = (h2 - h2.mean(0)) / (h2.std(0) + EPS)
        c   = z1.T @ z2 / N
        on_diag  = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = _off_diagonal(c).pow(2).sum()
        return on_diag + lambd * off_diag

    def forward(self, h_list):
        assert len(h_list) == self.n_encoders
        h_fusion = self.combine_att(h_list) if self.attention \
                   else self.combine_concat(h_list)
        semi = self.logistic(h_fusion).squeeze(0)

        loss_inter = 0.0
        for i in range(self.n_encoders):
            for j in range(i + 1, self.n_encoders):
                loss_inter = loss_inter + self._barlow_pair(
                    h_list[i], h_list[j], self.lambd_inter
                )
        return loss_inter, semi
