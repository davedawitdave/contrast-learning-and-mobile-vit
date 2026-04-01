# Contrastive Learning from Scratch on CIFAR-10

**Exploring Representation Geometry, Model Architectures, and Failure Modes in Self-Supervised Learning**

A clean, modular, from-scratch implementation of contrastive learning (SimCLR-style) on CIFAR-10. The project trains three encoders — SimpleCNN, ResNet-18, and EfficientNet-B0 — and deeply analyzes the learned embedding spaces using alignment, uniformity, clustering metrics, k-NN evaluation, confusion matrices, and controlled failure experiments.

---

###  Project Goals

- Understand **how neural networks learn visual similarity** without labels.
- Compare three very different architectures (vanilla CNN vs residual vs modern compound-scaled).
- Quantify representation quality with rigorous geometry and downstream metrics.
- Deliberately break the model in Notebook 3 to develop mechanistic intuition about collapse, temperature sensitivity, and hard negatives.
- Build production-grade habits: Google Drive checkpointing, AMP, clean modular notebooks, and comprehensive evaluation.


---

###  How to Run

1. Open any notebook in **Google Colab** (GPU runtime recommended).
2. Mount Google Drive (the code does this automatically).
3. Run cells sequentially.
4. All models and logs are saved to `/content/drive/MyDrive/contrastive_models/`.

**Note:** Notebook 1 takes ~1–2 hours on a T4 GPU (EfficientNet-B0 uses AMP and a smaller batch size). You can reduce `epochs=10` for faster experiments.

---

**Summary of Results**  
EfficientNet-B0 (with 32→224 resize + ImageNet initialization) clearly outperforms the other two models across **every metric**:
- Highest k-NN accuracy: **82.5%**
- Best cluster separation (Silhouette 0.188, Davies-Bouldin 2.778)
- Strongest positive similarity (0.290) while maintaining excellent uniformity

ResNet-18 is a solid middle performer. SimpleCNN works as a baseline but shows clear limitations in clustering quality.

---

### Technical Highlights

- **Augmentations**: Strong SimCLR-style pipeline (RandomResizedCrop, ColorJitter, GaussianBlur, Grayscale).
- **Loss**: Custom InfoNCE with temperature τ = 0.5.
- **Architectures**: SimpleCNN (baseline), ResNet-18 (CIFAR-adapted), EfficientNet-B0 (with internal resize).
- **Projection Head**: Non-linear MLP (critical for preventing collapse).
- **Training**: AdamW + Cosine Annealing + Automatic Mixed Precision (AMP) for EfficientNet-B0.
- **Evaluation**: Alignment, Uniformity, Silhouette, Davies-Bouldin, Intra/Inter ratio, k-NN (k=20), confusion matrices, cosine similarity histograms, and projection-head ablation.

- Temperature sensitivity analysis (τ = 0.01, 0.5, 10.0)
- Hard-negative distribution study
- Deliberate representation collapse simulation
- Full per-class confusion matrix analysis and failure-mode takeaways

```bash
torch torchvision torchaudio
scikit-learn
seaborn matplotlib pandas numpy tqdm
