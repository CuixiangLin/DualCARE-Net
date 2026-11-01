# DualCARE-Net

This repository contains the data and code of DualCARE-Net (Dual-branch Complementary
Aggregation Network), which learns complementary
nonlinear and linear representations from the same biological evidence through two
parallel branches of five feature encoders, and reduces redundancy among them with a
pairwise Barlow Twins objective. On the pan-cancer benchmark (796 positives and 2,187
negatives over 13,627 genes) it reaches AUPRC ~ 0.94 and AUROC ~ 0.97 under repeated
stratified 5-fold cross-validation.

## Author

Meng-Long Wang, Hong-Dong Li, Cui-Xiang Lin.

## Contact

Cui-Xiang Lin (lincx@xtu.edu.cn)

## Requirements

```bash
git clone https://github.com/CuixiangLin/DualCARE-Net.git
cd DualCARE-Net
pip install -r requirements.txt
```

Tested environment:
> python == 3.10.13 \
> torch == 2.4.0+cu121 (CUDA 12.1) \
> torch_geometric == 2.8.0 \
> numpy == 2.2.6 \
> pandas == 2.3.3 \
> scipy == 1.15.3 \
> scikit-learn == 1.7.2

Hardware used:
> GPU: NVIDIA A100-SXM4-80GB (driver 550.54.15, CUDA 12.4) \
> CPU: Intel Xeon Platinum 8358 @ 2.60GHz \
> OS: Ubuntu 22.04.4 LTS

A CUDA GPU is recommended: the **S** encoder builds a 13,627 x 13,627 dense PPI matrix,
which takes about 708 MiB of GPU memory.

## Data

`data/` and `processed_data/` hold the gene list, the positive and negative gene labels,
the MSigDB c2/c5 gene-set matrices, the cancer-related molecular features, and the confidence-weighted
STRING PPI network. `geneset_features_21288.txt.gz` is the full c2+c5 pool; the
disease-keyword filter is applied when it is loaded, leaving the 20,647 gene sets reported
in the paper.

## Usage

```bash
python main.py
```

The default configuration is the full **HCMBS** model, so this single command reproduces
the pan-cancer result; predictions are written to `results/pred_HCMBS_best.tsv`. Use
`--encoders` to run a subset of the five encoders, for example `--encoders H,C`. All
hyper-parameters, including the values used in the paper, are in `src/config.py`.

## Output

`results/pred_HCMBS_best.tsv` ranks all 13,627 genes by predicted score. The file shipped
here is the output of the run reported in the paper, so its ranks and scores match the
supplementary tables; retraining reproduces the ranking closely but not identically,
because GPU training is not bit-wise deterministic. The console log reports per-fold and
mean +/- std AUROC and AUPRC.
