
# STTFN:  Spatio-Temporal Tensor Fusion Network 

Official source code for the paper: **"Decoupled Representation Learning for Traffic Forecasting with Dual-Plane Spatial-Temporal Tensor Fusion"**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
The **STTFN** model introduces a dual-plane architecture that decouples spatial dependencies and temporal patterns into parallel branches (**SRGCN** and **AutoTRT**), which are then integrated using a learnable **Concatenation-based Tensor Fusion** strategy to achieve state-of-the-art results on traffic flow prediction tasks.

## 🏗️ Model Architecture
Our framework consists of:
- **Spatial Branch**: Adaptive S-R GCN (SRGCN) for capturing dynamic spatial dependencies with spectral-refined graph convolutions.
- **Temporal Branch**: Automated Temporal Representative Transformer (AutoTRT) for multi-scale temporal pattern extraction.
- **Fusion Module**: Advanced learnable fusion mechanism beyond simple addition or multiplication.

## 🚀 Installation
```bash
git clone https://github.com/InterpreterShi/STTFN.git
cd STTFN
pip install -r requirements.txt
```

## 📊 Dataset Preparation
We evaluate on **PeMS04, and **PeMS08**.
Place your `.npz` files in the `dataset/` directory as follows:
```
STTFN/
├── dataset/
│   ├── pems04/
│   │   └── pems04.npz
│   └── pems08/
│       └── pems08.npz
└── ...
```

## 🛠️ Quick Start
### Training
To train the model on a specific dataset (e.g., PeMS04):
```bash
python main_sttfn.py --dataset pems04 --device cuda:0
```

### Fusion Strategy Ablation
To reproduce the fusion strategy comparison results presented in Section 4.5 of our study:
```bash
python run_ablation.py --variant all --datasets pems04 pems08 --device cuda:0
```
Then use `visualize_ablation.py` to generate the comparison plots:
```bash
python visualize_ablation.py
```

## 📉 Experimental Results (Ablation)
Our ablation study on fusion methods (Section 4.5) demonstrates the superiority of **Concatenation** over other strategies:

| Dataset | Metric | Hadamard | Summation | **Concatenation** |
| :--- | :--- | :---: | :---: | :---: |
| PeMS04 (9-step) | MAE / RMSE | 32.96 / 47.39 | 29.00 / 42.58 | **28.73 / 42.29** |
| PeMS08 (9-step) | MAE / RMSE | 27.72 / 39.25 | 25.05 / 36.21 | **24.50 / 35.49** |

For more detailed results, check `experiment_results.json` and `fusion_ablation_results.png`.


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
