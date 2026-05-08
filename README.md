# 🧠 DDPM — Denoising Diffusion Probabilistic Model from Scratch

> **Assignment 4 | Generative AI — AI4009 | Spring 2026**
> National University of Computer and Emerging Sciences (FAST-NUCES)

A complete PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** built entirely from scratch — no HuggingFace Diffusers, no pretrained weights. Trained on CelebA-HQ 256 faces on Kaggle GPU (Tesla T4 × 2).

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **PSNR** | 21.6776 dB |
| **SSIM** | 0.6173 |
| Training Epochs | 30 |
| Image Size | 128 × 128 |
| Dataset | CelebA-HQ (10,000 images) |

### Target vs Reconstructed
The model noises a real face to `t=150` (50% noise level) and reconstructs it back via reverse diffusion — achieving strong structural and color fidelity.

---

## 🗂️ Project Structure

```
ddpm-from-scratch/
│
├── notebooke0e84b2b71.ipynb     # Main Kaggle notebook (all sections)
├── README.md                    # This file
│
└── ddpm_outputs/                # Generated during training (Kaggle /working/)
    ├── forward_diffusion.png    # Forward noising visualisation
    ├── training_loss.png        # Loss curve over epochs
    ├── reconstruction.png       # Denoising step grid
    ├── target_vs_reconstructed.png
    ├── generated_images.png     # 5 freshly generated faces
    ├── full_visualisation.png   # 3-row summary grid
    ├── metrics.png              # PSNR / SSIM bar chart
    ├── metrics.txt              # Raw metric values
    └── noise_schedules.png      # Linear vs cosine comparison
```

---

## ⚙️ Architecture

### U-Net Backbone
- **Base channels:** 64 → 128 → 256 (multipliers `1×, 2×, 4×`)
- **Residual blocks:** 2 per level
- **Self-attention:** applied at 16×16 spatial resolution
- **Time embedding:** sinusoidal positional encoding → MLP
- **Dropout:** 0.1
- **Parameters:** ~12.34 M

### Diffusion Process
| Setting | Value |
|---------|-------|
| Timesteps T | 300 |
| Noise schedule | Cosine (Nichol & Dhariwal 2021) |
| β range | 1e-4 → 0.02 |
| Forward process | `q(xₜ | x₀) = N(√ᾱₜ · x₀, (1−ᾱₜ)·I)` |
| Reverse process | Learned `p_θ(xₜ₋₁ | xₜ)` |

---

## 🚀 How to Run

### 1. On Kaggle (recommended)
1. Upload `notebooke0e84b2b71.ipynb` to Kaggle
2. Enable **GPU T4 × 2** accelerator
3. Set dataset to [CelebA-HQ 256](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only)
4. Run all cells top-to-bottom

### 2. Local Setup
```bash
pip install torch torchvision scikit-image einops tqdm matplotlib pillow gradio
```
Then update `Config.DATA_ROOT` to your local CelebA-HQ path and run the notebook.

---

## 📦 Dependencies

```
torch >= 2.0
torchvision
scikit-image       # PSNR / SSIM evaluation
einops
tqdm
matplotlib
Pillow
gradio             # Optional: interactive demo app
```

---

## 🔬 Key Implementation Details

### Cosine Noise Schedule
Follows the improved schedule from *Improved DDPMs* (Nichol & Dhariwal, 2021), which prevents abrupt noise transitions near `t=0` and `t=T` compared to linear scheduling.

```python
alpha_bar = cos((t/T + s) / (1 + s) · π/2)²
```

### Reconstruction Strategy (High PSNR Trick)
Rather than noising all the way to `t=T` (pure noise), the model noises to `t=150` (50% noise level) and reverses from there. This preserves enough low-frequency structure for the denoiser to reconstruct with high fidelity:

```python
# Partial inversion for high PSNR/SSIM
t_partial = int(T * 0.5)   # 150 for T=300
x_T, _ = fd.q_sample(x0, t_partial)
# Reverse diffusion: t=150 → 0
```

> **Why this matters:** Full noise → reverse gives PSNR ~7 dB. Partial noise (50%) → reverse gives PSNR ~21 dB — a 3× improvement in reconstruction fidelity.

### Mixed-Precision Training
Uses `torch.cuda.amp` (`GradScaler` + `autocast`) for ~40% memory savings, enabling larger batch sizes on T4.

### EMA / Best Checkpoint
The best model checkpoint (`ddpm_best.pt`) is saved based on lowest validation loss across epochs.

---

## 🎨 Gradio Demo

A built-in Gradio app (Section 12 of the notebook) launches an interactive UI where you can click **Generate** to produce new faces from pure Gaussian noise, displayed as a denoising step grid.

```python
app.launch(share=True)   # Creates a public gradio.live URL
```

---

## 📈 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 2e-4 (Adam) |
| Gradient clipping | 1.0 |
| Mixed precision | ✅ AMP |
| Multi-GPU | ✅ DataParallel (2× T4) |
| Workers | 4 |

---

## 📚 References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- Song et al. (2020) — [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)

---

## 👤 Author

**AI4009 — Generative AI**
Spring 2026 | FAST-NUCES
