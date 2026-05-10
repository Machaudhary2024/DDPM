"""
streamlit_app.py — DDPM Image Generator & Reconstructor
Run: streamlit run streamlit_app.py
"""

import os, math, io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDPM — Face Generator",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIG  (must match training config)
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    IMAGE_SIZE = 128
    CHANNELS   = 3
    T          = 300
    BETA_START = 1e-4
    BETA_END   = 0.02
    SCHEDULE   = "cosine"
    BASE_CH    = 64
    CH_MULT    = (1, 2, 4)
    NUM_RES    = 2
    ATTN_RES   = (16,)
    DROPOUT    = 0.1
    CKPT_PATH  = "C:\\ddpm_best.pt"   # put your checkpoint file here

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  NOISE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def make_beta_schedule(schedule, T, beta_start, beta_end):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = 1 - (ab[1:] / ab[:-1])
    return betas.clamp(0, 0.999)


# ─────────────────────────────────────────────────────────────────────────────
#  FORWARD DIFFUSION
# ─────────────────────────────────────────────────────────────────────────────
class ForwardDiffusion:
    def __init__(self, cfg, device):
        self.T = cfg.T
        self.device = device
        betas = make_beta_schedule(cfg.SCHEDULE, cfg.T, cfg.BETA_START, cfg.BETA_END)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = F.pad(ab[:-1], (1, 0), value=1.0)

        def r(x): return x.float().to(device)
        self.betas = r(betas)
        self.alphas = r(alphas)
        self.alpha_bar = r(ab)
        self.alpha_bar_prev = r(ab_prev)
        self.sqrt_ab = r(ab.sqrt())
        self.sqrt_1m_ab = r((1 - ab).sqrt())
        self.sqrt_recip_ab = r((1 / ab).sqrt())
        self.sqrt_recip_m1_ab = r((1 / ab - 1).sqrt())
        self.posterior_var = r(betas * (1 - ab_prev) / (1 - ab))
        self.posterior_log_var_clip = r(torch.log(self.posterior_var.clamp(min=1e-20)))
        self.posterior_mean_c1 = r(betas * ab_prev.sqrt() / (1 - ab))
        self.posterior_mean_c2 = r((1 - ab_prev) * alphas.sqrt() / (1 - ab))

    @staticmethod
    def _extract(arr, t, shape):
        return arr[t].reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._extract(self.sqrt_ab, t, x0.shape)
        s2 = self._extract(self.sqrt_1m_ab, t, x0.shape)
        return s1 * x0 + s2 * noise, noise

    def predict_x0(self, xt, t, eps):
        c1 = self._extract(self.sqrt_recip_ab, t, xt.shape)
        c2 = self._extract(self.sqrt_recip_m1_ab, t, xt.shape)
        return c1 * xt - c2 * eps

    def q_posterior(self, x0, xt, t):
        m = (self._extract(self.posterior_mean_c1, t, xt.shape) * x0
           + self._extract(self.posterior_mean_c2, t, xt.shape) * xt)
        v = self._extract(self.posterior_var, t, xt.shape)
        lv = self._extract(self.posterior_log_var_clip, t, xt.shape)
        return m, v, lv

    @torch.no_grad()
    def p_sample(self, model, xt, t_int):
        t_tensor = torch.full((xt.shape[0],), t_int, device=self.device, dtype=torch.long)
        eps = model(xt, t_tensor)
        x0_pred = self.predict_x0(xt, t_tensor, eps).clamp(-1, 1)
        mean, _, log_var = self.q_posterior(x0_pred, xt, t_tensor)
        if t_int == 0:
            return mean
        noise = torch.randn_like(xt)
        return mean + (0.5 * log_var).exp() * noise


# ─────────────────────────────────────────────────────────────────────────────
#  U-NET COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmb(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(SinusoidalPE(ch), nn.Linear(ch, ch * 4),
                                 nn.SiLU(), nn.Linear(ch * 4, ch * 4))

    def forward(self, t):
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_ch, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_ch, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class UNetClean(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base  = cfg.BASE_CH
        mults = cfg.CH_MULT
        chs   = [base * m for m in mults]
        t_ch  = base * 4
        dr    = cfg.DROPOUT
        self.time_emb = TimeEmb(base)
        self.init_conv = nn.Conv2d(cfg.CHANNELS, base, 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.enc_downs  = nn.ModuleList()
        skip_chs = [base]
        in_ch = base
        res = cfg.IMAGE_SIZE
        for ch in chs:
            for _ in range(cfg.NUM_RES):
                self.enc_blocks.append(ResBlock(in_ch, ch, t_ch, dr))
                if res in cfg.ATTN_RES:
                    self.enc_blocks.append(AttnBlock(ch))
                else:
                    self.enc_blocks.append(nn.Identity())
                skip_chs.append(ch)
                in_ch = ch
            self.enc_downs.append(nn.Conv2d(ch, ch, 4, 2, 1))
            skip_chs.append(ch)
            res //= 2

        self.mid1 = ResBlock(in_ch, in_ch, t_ch, dr)
        self.mid_attn = AttnBlock(in_ch)
        self.mid2 = ResBlock(in_ch, in_ch, t_ch, dr)

        self.dec_blocks = nn.ModuleList()
        self.dec_ups    = nn.ModuleList()
        res = cfg.IMAGE_SIZE // (2 ** len(chs))
        for ch in reversed(chs):
            self.dec_ups.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(in_ch, ch, 3, padding=1)))
            in_ch = ch + skip_chs.pop()
            for _ in range(cfg.NUM_RES):
                self.dec_blocks.append(ResBlock(in_ch, ch, t_ch, dr))
                if res in cfg.ATTN_RES:
                    self.dec_blocks.append(AttnBlock(ch))
                else:
                    self.dec_blocks.append(nn.Identity())
                in_ch = ch + skip_chs.pop()
            res *= 2
        self.final = nn.Sequential(nn.GroupNorm(8, in_ch - skip_chs[-1] if skip_chs else in_ch),
                                   nn.SiLU(), nn.Conv2d(in_ch, cfg.CHANNELS, 3, padding=1))
        self._last_skip = skip_chs

    def forward(self, x, t):
        te = self.time_emb(t)
        h  = self.init_conv(x)
        skips = [h]
        bi = 0
        for di in range(len(self.enc_downs)):
            for _ in range(cfg.NUM_RES):
                h = self.enc_blocks[bi](h, te) if isinstance(self.enc_blocks[bi], ResBlock) else h
                bi += 1
                h = self.enc_blocks[bi](h) if not isinstance(self.enc_blocks[bi], nn.Identity) else h
                bi += 1
                skips.append(h)
            h = self.enc_downs[di](h)
            skips.append(h)
        h = self.mid1(h, te)
        h = self.mid_attn(h)
        h = self.mid2(h, te)
        bi2 = 0
        for ui in range(len(self.dec_ups)):
            h = h + skips.pop() if h.shape == skips[-1].shape else self.dec_ups[ui](h)
            h = self.dec_ups[ui](h) if ui < len(self.dec_ups) else h
            for _ in range(cfg.NUM_RES):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.dec_blocks[bi2](h, te) if isinstance(self.dec_blocks[bi2], ResBlock) else h
                bi2 += 1
                h = self.dec_blocks[bi2](h) if not isinstance(self.dec_blocks[bi2], nn.Identity) else h
                bi2 += 1
        return self.final(h)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tensor_to_pil(t):
    """[-1,1] tensor (C,H,W) → PIL Image."""
    t = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def grid_to_pil(tensors):
    """List of (C,H,W) tensors → single PIL grid image."""
    stack = torch.stack(tensors)
    grid  = make_grid(stack, nrow=len(tensors), normalize=True, value_range=(-1, 1))
    return tensor_to_pil(grid)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    fd  = ForwardDiffusion(cfg, DEVICE)
    net = UNetClean(cfg).to(DEVICE)
    if os.path.exists(cfg.CKPT_PATH):
        net.load_state_dict(torch.load(cfg.CKPT_PATH, map_location=DEVICE))
        net.eval()
        return fd, net, True
    return fd, net, False


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_images(fd, net, n_images=1, n_steps=8):
    shape  = (n_images, cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    cap_ts = sorted({int(cfg.T * i / (n_steps - 1)) for i in range(n_steps)} | {0}, reverse=True)
    x = torch.randn(shape, device=DEVICE)
    frames = []
    bar = st.progress(0, text="Generating…")
    for idx, t in enumerate(reversed(range(cfg.T))):
        x = fd.p_sample(net, x, t)
        if t in cap_ts:
            frames.append(x[0].cpu().clone())
        bar.progress((idx + 1) / cfg.T, text=f"Step {cfg.T - t}/{cfg.T}")
    bar.empty()
    return x[0].cpu(), frames


# ─────────────────────────────────────────────────────────────────────────────
#  RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct_image(fd, net, img_tensor, noise_level=0.5, n_steps=8):
    x0       = img_tensor.unsqueeze(0).to(DEVICE)
    t_partial = max(1, int(cfg.T * noise_level) - 1)
    t_tensor  = torch.tensor([t_partial], device=DEVICE)
    x_T, _   = fd.q_sample(x0, t_tensor)
    cap_ts    = sorted({int(t_partial * i / (n_steps - 1)) for i in range(n_steps)} | {0}, reverse=True)
    x = x_T.clone()
    frames = [x_T[0].cpu().clone()]
    bar = st.progress(0, text="Reconstructing…")
    total = t_partial + 1
    for idx, t in enumerate(reversed(range(t_partial + 1))):
        x = fd.p_sample(net, x, t)
        if t in cap_ts:
            frames.append(x[0].cpu().clone())
        bar.progress((idx + 1) / total, text=f"Step {t_partial - t}/{t_partial}")
    bar.empty()
    return x[0].cpu(), x_T[0].cpu(), frames


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid #2a2a4a;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    color: #e0e0ff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -1px;
}
.hero p { color: #8888bb; margin: 0; font-size: 1rem; }
.badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
    border: 1px solid #2563eb44;
}
.metric-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #60a5fa;
    font-weight: 700;
}
.metric-lbl { color: #64748b; font-size: 0.8rem; margin-top: 4px; }
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #60a5fa;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
  <div style="margin-bottom:1rem">
    <span class="badge">Generative AI</span>
  </div>
  <h1>DDPM Face Generator</h1>
  <p>Denoising Diffusion Probabilistic Model — trained from scratch on CelebA-HQ &nbsp;·&nbsp; Pure PyTorch</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-box"><div class="metric-val">21.68</div><div class="metric-lbl">PSNR (dB)</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box"><div class="metric-val">0.617</div><div class="metric-lbl">SSIM Score</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box"><div class="metric-val">12.3M</div><div class="metric-lbl">Parameters</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-box"><div class="metric-val">300</div><div class="metric-lbl">Diffusion Steps</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load model
fd, net, ckpt_loaded = load_model()

if not ckpt_loaded:
    st.warning(
        " No checkpoint found at `ddpm_best.pt`. "
        "Download it from your Kaggle output and place it in the same folder as this file. "
        "The app will still run but will produce random (untrained) outputs.",
        icon="⚠️"
    )

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Generate New Faces", " Reconstruct an Image"])

# ── TAB 1: GENERATE ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-label">Generation Settings</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_gen   = st.slider("Number of images", 1, 4, 1)
        n_vis   = st.slider("Visualisation steps", 4, 12, 8)
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button(" Generate", use_container_width=True, type="primary")

    with col_b:
        st.info("Click **Generate** to run reverse diffusion from pure Gaussian noise. "
                "The denoising grid shows intermediate steps from noise → final face.")

    if gen_btn:
        final, frames = generate_images(fd, net, n_images=1, n_steps=n_vis)
        st.markdown('<p class="section-label">Denoising Steps</p>', unsafe_allow_html=True)
        cols = st.columns(len(frames))
        for i, (col, frame) in enumerate(zip(cols, frames)):
            t_val = cfg.T - int(cfg.T * i / (len(frames) - 1)) if len(frames) > 1 else 0
            col.image(tensor_to_pil(frame), caption=f"t={t_val}", use_container_width=True)
        st.markdown('<p class="section-label">Final Generated Image</p>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([1, 2, 1])
        with fc2:
            st.image(tensor_to_pil(final), use_container_width=True)
        buf = io.BytesIO()
        tensor_to_pil(final).save(buf, format="PNG")
        st.download_button(" Download Image", buf.getvalue(), "generated_face.png", "image/png")

# ── TAB 2: RECONSTRUCT ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Reconstruction Settings</p>', unsafe_allow_html=True)
    col_c, col_d = st.columns([1, 2])
    with col_c:
        noise_level = st.slider(
            "Noise level", 0.2, 0.8, 0.5, 0.05,
            help="Fraction of T to noise to. Lower = higher PSNR. 0.5 ≈ 21 dB."
        )
        n_vis_r = st.slider("Visualisation steps", 4, 10, 6)
        uploaded = st.file_uploader("Upload a face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    with col_d:
        st.info(
            f"The image is noised to **t = {int(cfg.T * noise_level)}** "
            f"({int(noise_level*100)}% of T={cfg.T}), then reconstructed via reverse diffusion. "
            "Lower noise → higher fidelity; higher noise → more model creativity."
        )

    if uploaded is not None:
        raw_img = Image.open(uploaded).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.CenterCrop(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        img_tensor = transform(raw_img)
        rec_btn = st.button(" Reconstruct", use_container_width=True, type="primary")

        if rec_btn:
            recon, x_T, frames = reconstruct_image(fd, net, img_tensor, noise_level, n_vis_r)

            st.markdown('<p class="section-label">Side-by-Side Comparison</p>', unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            ca.image(tensor_to_pil(img_tensor), caption=" Target (original)", use_container_width=True)
            cb.image(tensor_to_pil(x_T),        caption=f" Noised (t={int(cfg.T*noise_level)})", use_container_width=True)
            cc.image(tensor_to_pil(recon),       caption=" Reconstructed",    use_container_width=True)

            st.markdown('<p class="section-label">Denoising Steps</p>', unsafe_allow_html=True)
            step_cols = st.columns(len(frames))
            for i, (col, frame) in enumerate(zip(step_cols, frames)):
                col.image(tensor_to_pil(frame), caption=f"Step {i}", use_container_width=True)

            # PSNR / SSIM
            try:
                from skimage.metrics import peak_signal_noise_ratio as psnr_fn
                from skimage.metrics import structural_similarity as ssim_fn
                t_np = ((img_tensor.clamp(-1,1)+1)/2).permute(1,2,0).numpy().astype("float32")
                r_np = ((recon.clamp(-1,1)+1)/2).permute(1,2,0).numpy().astype("float32")
                psnr = psnr_fn(t_np, r_np, data_range=1.0)
                ssim = ssim_fn(t_np, r_np, data_range=1.0, channel_axis=2)
                st.markdown('<p class="section-label">Metrics for this reconstruction</p>', unsafe_allow_html=True)
                pm, sm = st.columns(2)
                pm.markdown(f'<div class="metric-box"><div class="metric-val">{psnr:.2f}</div><div class="metric-lbl">PSNR (dB)</div></div>', unsafe_allow_html=True)
                sm.markdown(f'<div class="metric-box"><div class="metric-val">{ssim:.4f}</div><div class="metric-lbl">SSIM</div></div>', unsafe_allow_html=True)
            except ImportError:
                pass

            buf2 = io.BytesIO()
            tensor_to_pil(recon).save(buf2, format="PNG")
            st.download_button(" Download Reconstruction", buf2.getvalue(), "reconstructed.png", "image/png")
    else:
        st.markdown("Upload an image above to get started.")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Model Info")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Checkpoint:** {'Loaded' if ckpt_loaded else ' Not found'}")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(f"- Image size: `{cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE}`")
    st.markdown(f"- Timesteps T: `{cfg.T}`")
    st.markdown(f"- Schedule: `{cfg.SCHEDULE}`")
    st.markdown(f"- Base channels: `{cfg.BASE_CH}`")
    st.markdown(f"- Attention res: `{cfg.ATTN_RES}`")
    st.markdown("---")
    st.markdown("**DDPM**")
    st.markdown("GenAI")
    st.markdown("Mehar Akbar")
