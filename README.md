# Contrastive Learning from Scratch on CIFAR-10
**Exploring Representation Geometry, Model Architectures, and Failure Modes in Self-Supervised Learning**

A clean, modular, from-scratch implementation of contrastive learning (SimCLR-style) on CIFAR-10. The project trains three encoders — SimpleCNN, ResNet-18, and EfficientNet-B0 — and deeply analyzes the learned embedding spaces using alignment, uniformity, clustering metrics, k-NN evaluation, confusion matrices, and controlled failure experiments.

---

### Project Goals
- Understand **how neural networks learn visual similarity** without any labels.
- Compare three very different architectures (vanilla CNN vs residual vs modern compound-scaled).
- Quantify representation quality with rigorous geometry and downstream metrics.
- Perform controlled experiments on weak vs strong augmentation and temperature sensitivity to develop mechanistic intuition about collapse and failure modes.
- Build production-grade habits: Google Drive checkpointing, AMP, clean modular notebooks, and comprehensive evaluation.

---

### Repository Structure
```
contrastive-cifar10/
├── 01_Combined_Data_Models_Training_Initial_Embeddings.ipynb
├── 02_Representation_Geometry_and_Evaluation.ipynb
├── 03_Experiments_and_Failure_Analysis.ipynb
├── images/                          # All result plots (training curves, t-SNE, PCA, metrics, confusion matrices)
├── README.md
└── requirements.txt
```

---

### How to Run
1. Open any notebook in **Google Colab** (GPU runtime recommended).
2. Mount Google Drive (the code does this automatically).
3. Run cells sequentially.
4. All models and logs are saved to `/content/drive/MyDrive/contrastive_models/` (main runs) and `/experiments/` (controlled experiments).

**Note:** Notebook 1 takes ~1–2 hours on a T4 GPU (EfficientNet-B0 uses AMP and smaller batch size). You can reduce `epochs=10` for faster experiments.

---

### Summary of Results

**Main Training (Strong Augmentation)**  
EfficientNet-B0 (with 32→224 resize + ImageNet initialization) clearly outperforms the other two models across **every metric**:
- Highest k-NN accuracy: **82.5%**
- Best cluster separation (Silhouette 0.188, Davies-Bouldin 2.778)
- Strongest positive similarity (0.290) while maintaining excellent uniformity

ResNet-18 is a solid middle performer. SimpleCNN works as a baseline but shows clear limitations in clustering quality.

**Controlled Experiments (Notebook 3)**  
We trained EfficientNet-B0 with **only weak augmentation** (10 epochs) under two temperatures:

- **τ = 0.5**: Better alignment and cluster separation  
- **τ = 0.1**: Lower positive similarity, more overlap in t-SNE/PCA, and reduced k-NN performance

**Key Insight**: Strong data augmentation is more critical than temperature tuning. Weak augmentations severely degrade representation quality even when temperature is varied.

### Technical Highlights
- **Augmentations**: Strong SimCLR-style (RandomResizedCrop, ColorJitter, GaussianBlur, Grayscale) vs Weak (only RandomHorizontalFlip).
- **Loss**: Custom InfoNCE with temperature τ (default 0.5, ablation 0.1).
- **Architectures**: SimpleCNN (baseline), ResNet-18 (CIFAR-adapted), EfficientNet-B0 (with internal resize).
- **Projection Head**: Non-linear MLP (critical for preventing collapse).
- **Training**: AdamW + Cosine Annealing + Automatic Mixed Precision (AMP) for EfficientNet-B0.
- **Evaluation**: Alignment (lower = better), Uniformity, Silhouette, Davies-Bouldin, Intra/Inter ratio, k-NN (k=20), confusion matrices, cosine similarity histograms, t-SNE/PCA, and projection-head ablation.
- **Controlled Experiments**: Weak vs strong augmentation and temperature sensitivity.

---

### Requirements
```bash
torch torchvision torchaudio
scikit-learn
seaborn matplotlib pandas numpy tqdm
```

(Install via `pip install -r requirements.txt`)

---

**Built as a teaching / research tool to develop deep mechanistic understanding of contrastive learning.**

If you find this useful, please ⭐ the repo!

---
```
Just create an `images/` folder in your repo and place the six images you shared with the filenames I used above (or update the links if you prefer different names).

Would you like me to also generate the `requirements.txt` file or a short `LICENSE`? Just say the word!
