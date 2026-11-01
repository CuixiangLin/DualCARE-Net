# -*- coding: utf-8 -*-
"""Central configuration: paths and hyper-parameters for DualCARE-Net."""
import os

# ============== Paths ==============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
PROC_DIR = os.path.join(ROOT, 'processed_data')

GENE_LIST_PATH = os.path.join(DATA_DIR, 'geneList.txt')
POSITIVE_PATH  = os.path.join(DATA_DIR, '796true.txt')
NEGATIVE_PATH  = os.path.join(DATA_DIR, '2187false.txt')

MULTIOMICS_FEATURE_PATH = os.path.join(PROC_DIR, 'multiomics_features_98dim.csv')
# Binary c2/c5 gene-set membership features used by the B encoder. The file is the
# gzip-compressed c2+c5 pool (pandas decompresses it by extension); the
# disease-keyword filter is applied when it is loaded, leaving the same 20,647
# gene sets the H encoder uses.
GENESET_FEATURE_PATH = os.path.join(PROC_DIR, 'geneset_features_21288.txt.gz')

# Confidence-weighted PPI network: STRING v12.0 human interactions with a
# normalized combined score above 0.5, supplemented with CPDB edges for the genes
# left isolated by that filter. 453,092 edges covering all 13,627 genes.
PPI_PATHS = {
    'string_conf': os.path.join(PROC_DIR, 'ppi_string_conf_weighted.tsv'),
}

# ============== Model hyper-parameters ==============
FEATURE_DIM = 98
N_HID       = 128
DROPOUT     = 0.3
N_CLASS     = 2
N_LAYERS    = 3
CHEB_K      = 2

# ============== Fusion loss ==============
LAMBDA_INTER  = 5e-3       # off-diagonal weight inside the Barlow Twins term
W_SELF        = 0.001      # weight of the redundancy-reduction term in the total loss
USE_ATTENTION = False      # False = concatenation, True = additive attention

# ============== PPI edge weighting ==============
# w_ij = (1 - ALPHA) * conf_ij + ALPHA * mutcos(i, j)
# ALPHA = 0 reduces this to the confidence-weighted PPI, which is the pan-cancer
# setting reported in the paper. A value above 0 requires a PPI file carrying a
# 'mutcos' column; the file shipped here provides confidence scores only.
PPI_VERSION  = 'string_conf'
ALPHA        = 0.0

# ============== Optimizers ==============
LR_FUSION       = 1e-6
WD_FUSION       = 5e-5

# Per-encoder learning rates and weight decays
LR_H            = 1e-2     # H: hypergraph encoder
WD_H            = 5e-6
LR_C            = 5e-3     # C: Chebyshev graph encoder
WD_C            = 0.0
LR_M            = 5e-3     # M: molecular feature encoder
WD_M            = 0.0
LR_B            = 1e-4     # B: biological gene-set feature encoder
WD_B            = 1e-4
LR_S            = 5e-4     # S: weighted STRING PPI adjacency encoder
WD_S            = 1e-3

LR_MILESTONES   = [100, 200, 300, 400]
LR_GAMMA        = 0.5

# ============== Encoder selection ==============
# The full model is HCMBS. Any subset of 1-5 encoders can be chosen via --encoders.
# Valid keys (case-insensitive, comma-separated):
#   H = hypergraph encoder                      (gene-set hypergraph; nonlinear branch)
#   C = Chebyshev graph encoder                 (molecular features over the weighted PPI;
#                                                nonlinear branch)
#   M = molecular feature encoder               (98-dim features, no propagation; linear branch)
#   B = biological gene-set feature encoder     (binary c2/c5 features; linear branch)
#   S = weighted STRING PPI adjacency encoder   (PPI adjacency vectors; linear branch)
ENCODERS = 'H,C,M,B,S'

# ============== Training ==============
EPOCHS    = 200
N_REPEATS = 5              # repeats of the 5-fold cross-validation
N_SPLITS  = 5
SEED      = 123

# ============== Hypergraph ==============
HYPEREDGE_MIN_POSITIVE = 3
