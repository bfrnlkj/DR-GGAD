# DR-GGAD: Dual Residual Centering for Mitigating Anomaly Non‑Discriminativity in Generalist Graph Anomaly Detection

## Abstract
Generalist Graph Anomaly Detection (GGAD) seeks a unified representation learning model to detect anomalies in unseen graphs, but cross-domain transfer often entangles the learned anomalous and normal representations. We formalize this degradation as Anomaly non-Discriminativity (AnD) and define a normalized score to quantify it. We present DR-GGAD, which avoids direct comparison between anomalous and normal nodes via two residual modules: 1) a multi-scale Hyper Residual (HR) Center measuring node-to-center distances, yielding a compact normal residual structure with margin-pushed anomalies; 2) an Affinity-Residual (AR) module enforcing local residual directional consistency to recover structural separability. With frozen parameters (no target fine-tuning), DR-GGAD fuses both signals into a unified score. On 8 benchmark target graphs, it achieves new SOTA: mean AUROC +5.14% over the best prior GGAD, with large gains on high-AnD datasets (ACM +9.96%, Amazon +7.48%) and strong AUPRC boosts (Amazon +17.12%, CiteSeer +17.77%). Ablations confirm complementary roles of the two modules. DR-GGAD thus establishes AnD as a measurable bottleneck and delivers robust cross-domain anomaly detection.

## Dataset Preparation

We follow the same dataset splits and preprocessing as **[ARC (A Generalist Graph Anomaly Detector with In-Context Learning)](https://github.com/yixinliu233/ARC)**. Please refer to their repository for dataset download and preparation details.

## 🛠️ Requirements

### Environment Setup

```bash
# Create conda environment
conda create -n dr-ggad python=3.8.14
conda activate dr-ggad

# Install PyTorch with CUDA support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install DGL with CUDA support
pip install dgl==0.9.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

# Install other dependencies
pip install -r requirements.txt
