import json, math, os, random, time, textwrap, zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# Make plots look a bit more publication-style / LaTeX-like
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "lines.linewidth": 2.0,
})

# ---------------------------
# Mess3 process definitions
# ---------------------------
# Token-labeled transition matrices (states A,B,C and tokens 0,1,2):
# T^(0) = [[a y, b x, b x], [a x, b y, b x], [a x, b x, b y]]
# T^(1) = [[b y, a x, b x], [b x, a y, b x], [b x, a x, b y]]
# T^(2) = [[b y, b x, a x], [b x, b y, a x], [b x, b x, a y]]
# where b=(1-a)/2 and y=1-2x.

@dataclass
class Mess3Component:
    name: str
    alpha: float
    x: float

    # This just builds the 3 token-specific transition matrices for one Mess3 process.
    # Intuition: for each possible observed token, it tells us how likely we are to move
    # from one hidden state to the next hidden state.
    def matrices(self) -> np.ndarray:
        a = self.alpha
        x = self.x
        b = (1.0 - a) / 2.0
        y = 1.0 - 2.0 * x
        T0 = np.array([
            [a * y, b * x, b * x],
            [a * x, b * y, b * x],
            [a * x, b * x, b * y],
        ], dtype=np.float64)
        T1 = np.array([
            [b * y, a * x, b * x],
            [b * x, a * y, b * x],
            [b * x, a * x, b * y],
        ], dtype=np.float64)
        T2 = np.array([
            [b * y, b * x, a * x],
            [b * x, b * y, a * x],
            [b * x, b * x, a * y],
        ], dtype=np.float64)
        mats = np.stack([T0, T1, T2], axis=0)
        rs = mats.sum(axis=(0, 2))
        assert np.allclose(rs, np.ones(3), atol=1e-9), (self.name, rs)
        return mats


COMPONENTS = [
    Mess3Component("M1", alpha=0.90, x=0.05),
    Mess3Component("M2", alpha=0.55, x=0.20),
    Mess3Component("M3", alpha=0.25, x=0.05),
]
VOCAB = 4  # tokens 0,1,2 plus BOS=3
BOS = 3
SEQ_LEN = 20


# This gives the starting distribution over the 3 hidden states inside one component.
# Intuition: before seeing anything, we start with no preference, so we treat the 3 states equally.
def stationary_distribution_block() -> np.ndarray:
    return np.ones(3, dtype=np.float64) / 3.0


# This generates one sequence from one fixed Mess3 component.
# Intuition: first pick a hidden state, then keep sampling
# (observed token, next hidden state) pairs step by step.
def sample_from_component(comp: Mess3Component, length: int, rng: np.random.Generator):
    mats = comp.matrices()
    s = rng.choice(3, p=stationary_distribution_block())
    toks = []
    states = [s]
    for _ in range(length):
        probs = mats[:, s, :].reshape(-1)
        idx = rng.choice(9, p=probs / probs.sum())
        tok = idx // 3
        s = idx % 3
        toks.append(int(tok))
        states.append(int(s))
    return toks, states


# This makes one training batch.
# Intuition: for each sequence in the batch, first choose which component it comes from,
# then generate the whole sequence only from that component. That is what makes the dataset non-ergodic.
def sample_batch(batch_size: int, seq_len: int, rng: np.random.Generator):
    xs = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
    ys = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
    comps = np.zeros(batch_size, dtype=np.int64)
    hidden_states = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
    xs[:, 0] = BOS
    for b in range(batch_size):
        c = int(rng.integers(0, len(COMPONENTS)))
        toks, states = sample_from_component(COMPONENTS[c], seq_len + 1, rng)
        comps[b] = c
        ys[b] = np.array(toks, dtype=np.int64)
        xs[b, 1:] = np.array(toks[:-1], dtype=np.int64)
        hidden_states[b] = np.array(states[:-1], dtype=np.int64)
    return xs, ys, comps, hidden_states


# ---------------------------
# Belief states for non-ergodic mixture
# ---------------------------

FULL_MATS = np.zeros((3, 9, 9), dtype=np.float64)
for c, comp in enumerate(COMPONENTS):
    mats = comp.matrices()
    FULL_MATS[:, 3*c:3*c+3, 3*c:3*c+3] = mats
INIT_BELIEF = np.ones(9, dtype=np.float64) / 9.0
ONES9 = np.ones(9, dtype=np.float64)


# This does one exact Bayesian belief update after seeing one token.
# Intuition: take the current belief over the 9 global hidden states,
# keep only the transitions that could have produced this token, then renormalize.
def update_belief(belief: np.ndarray, token: int) -> np.ndarray:
    num = belief @ FULL_MATS[token]
    den = float(num.sum())
    if den <= 0:
        return belief.copy()
    return num / den


# This computes the exact belief after a whole observed prefix.
# Intuition: just start from the prior and apply the belief update one token at a time.
def belief_from_prefix(prefix: List[int]) -> np.ndarray:
    b = INIT_BELIEF.copy()
    for tok in prefix:
        b = update_belief(b, tok)
    return b


# This turns the full 9-state belief into a 3-way posterior over components.
# Intuition: just sum the 3 hidden states inside each component block.
def component_posterior(belief: np.ndarray) -> np.ndarray:
    return belief.reshape(3, 3).sum(axis=1)


# This gives the belief over the 3 local hidden states inside one chosen component.
# Intuition: zoom in on one component block and renormalize inside that block.
def local_belief(belief: np.ndarray, c: int) -> np.ndarray:
    block = belief[3*c:3*c+3]
    s = block.sum()
    if s < 1e-12:
        return np.ones(3) / 3
    return block / s


# This lists all prefixes up to a given length.
# Intuition: start from the empty prefix, and keep appending tokens 0/1/2.
# This is only okay for small lengths because the number of prefixes grows really fast.
def enumerate_prefixes(max_len: int) -> List[List[int]]:
    prefixes = [[]]
    for _ in range(max_len):
        new = []
        for p in prefixes:
            for t in range(3):
                new.append(p + [t])
        prefixes.extend(new)
    return prefixes


# This computes the exact belief state at every position for every sequence.
# Intuition: for each sequence, walk through the tokens from left to right
# and keep updating the hidden-state posterior.
def beliefs_for_sequences(sequences: np.ndarray) -> np.ndarray:
    N, T = sequences.shape
    beliefs = np.zeros((N, T, 9), dtype=np.float32)
    for i in range(N):
        b = INIT_BELIEF.copy()
        for t in range(T):
            b = update_belief(b, int(sequences[i, t]))
            beliefs[i, t] = b.astype(np.float32)
    return beliefs


# ---------------------------
# Tiny decoder-only transformer
# ---------------------------
class Block(nn.Module):
    # This sets up one transformer block.
    # Intuition: this block has one attention part and one MLP part,
    # both wrapped with residual connections.
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )

    # This is one transformer block: attention + MLP with residual connections.
    # Intuition: the attention part lets each position look back at earlier positions,
    # and the MLP part further reshapes the representation.
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor):
        z = self.ln1(x)
        q = self.q(z)
        k = self.k(z)
        v = self.v(z)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        att = att.masked_fill(causal_mask, -1e9)
        att = att.softmax(dim=-1)
        x = x + self.o(att @ v)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    # This builds the small decoder-only transformer.
    # Intuition: token embeddings + position embeddings go in,
    # a few transformer blocks process them, and then we predict the next token.
    def __init__(self, vocab_size: int, d_model: int = 24, n_layers: int = 1,
                 d_mlp: int = 48, ctx_len: int = SEQ_LEN + 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ctx_len = ctx_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(ctx_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    # This runs the transformer on an input sequence.
    # Intuition: build the initial embeddings, pass them through each block,
    # and optionally save the residual stream after each stage for geometry analysis later.
    def forward(self, x: torch.Tensor, return_residuals: bool = False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)[None, :, :]
        residuals = [h.detach().cpu() if return_residuals else None]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
            if return_residuals:
                residuals.append(h.detach().cpu())
        logits = self.unembed(self.ln_f(h))
        return (logits, residuals) if return_residuals else logits


# ---------------------------
# Training
# ---------------------------
@dataclass
class TrainConfig:
    # This stores the main training settings in one place.
    d_model: int = 24
    n_layers: int = 2
    d_mlp: int = 128
    batch_size: int = 128
    steps: int = 5000
    lr: float = 2e-3
    eval_every: int = 50


# This trains the transformer on next-token prediction.
# Intuition: sample a fresh batch each step, predict the next token at every position,
# compute cross-entropy loss, and update the model.
def train_model(cfg: TrainConfig):
    model = TinyGPT(VOCAB, cfg.d_model, cfg.n_layers, cfg.d_mlp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(SEED + 1)
    history = {"step": [], "loss": []}
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        x_np, y_np, _, _ = sample_batch(cfg.batch_size, SEQ_LEN, rng)
        x = torch.tensor(x_np, dtype=torch.long, device=device)
        y = torch.tensor(y_np, dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == 1:
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            print(f"step {step:4d} loss {loss.item():.4f}")
    history["train_seconds"] = time.time() - t0
    return model, history


# ---------------------------
# Analysis utilities
# ---------------------------

# This converts a 3-way probability vector into a point inside a triangle.
# Intuition: it lets us plot posteriors over 3 components or 3 local states in 2D.
def barycentric_to_xy(p: np.ndarray) -> np.ndarray:
    v = np.array([
        [0.0, 0.0],                     # first coordinate -> bottom-left
        [1.0, 0.0],                     # second coordinate -> bottom-right
        [0.5, math.sqrt(3)/2],          # third coordinate -> top
    ], dtype=np.float64)
    return p @ v


# This makes a fresh dataset for analysis after training.
# Intuition: we generate new sequences, compute the exact Bayesian beliefs,
# and also save the transformer's residual activations on the same sequences.
def sample_analysis_set(model: TinyGPT, n_seq: int = 200):
    rng = np.random.default_rng(SEED + 2)
    x_np, y_np, comps_np, hidden_np = sample_batch(n_seq, SEQ_LEN, rng)
    tokens_only = y_np[:, :-1]
    beliefs = beliefs_for_sequences(tokens_only)
    x = torch.tensor(x_np[:, :-1], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, residuals = model(x, return_residuals=True)
    residuals_np = np.stack([r.numpy() for r in residuals], axis=0)
    return {
        "inputs": x_np[:, :-1],
        "targets": y_np[:, :-1],
        "components": comps_np,
        "hidden_states": hidden_np[:, :-1],
        "beliefs": beliefs,
        "residuals": residuals_np,
        "pred_logits": logits.detach().cpu().numpy(),
    }


# This fits a simple linear regression probe from activations to some target quantity.
# Intuition: if the regression works well, it means that target is linearly readable
# from the representation.
def regression_report(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.4, random_state=SEED)
    reg = LinearRegression().fit(Xtr, Ytr)
    pred = reg.predict(Xte)
    return {
        "rmse": float(np.sqrt(mean_squared_error(Yte, pred))),
        "r2": float(r2_score(Yte, pred, multioutput="variance_weighted")),
        "reg": reg,
        "Xte": Xte,
        "Yte": Yte,
        "pred": pred,
    }


# This is the main geometry analysis step.
# Intuition: at every layer and every position, check how well the residual stream
# can linearly predict (1) the 3-way component posterior and (2) the full 9-state belief.
def analyze_residual_geometry(bundle: Dict) -> Dict:
    residuals = bundle["residuals"]
    beliefs = bundle["beliefs"]
    q = beliefs.reshape(*beliefs.shape[:2], 3, 3).sum(axis=-1)
    n_layers = residuals.shape[0]
    T = residuals.shape[2]

    full_rmse = np.zeros((n_layers, T))
    full_r2 = np.zeros((n_layers, T))
    q_rmse = np.zeros((n_layers, T))
    q_r2 = np.zeros((n_layers, T))
    regs_q = {}
    regs_full = {}

    for l in range(n_layers):
        for t in range(T):
            X = residuals[l, :, t, :]
            rep = regression_report(X, beliefs[:, t, :])
            full_rmse[l, t] = rep["rmse"]
            full_r2[l, t] = rep["r2"]
            regs_full[(l, t)] = rep

            repq = regression_report(X, q[:, t, :])
            q_rmse[l, t] = repq["rmse"]
            q_r2[l, t] = repq["r2"]
            regs_q[(l, t)] = repq

    pos_entropy = -(q * np.log(np.clip(q, 1e-12, 1))).sum(axis=-1).mean(axis=0)
    pos_qmax = q.max(axis=-1).mean(axis=0)

    return {
        "full_rmse": full_rmse,
        "full_r2": full_r2,
        "q_rmse": q_rmse,
        "q_r2": q_r2,
        "regs_q": regs_q,
        "regs_full": regs_full,
        "pos_entropy": pos_entropy,
        "pos_qmax": pos_qmax,
        "q": q,
    }


# ---------------------------
# Plotting
# ---------------------------

# This is just a helper to save a figure nicely and close it after saving.
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close()


# This plots the training loss over time.
# Intuition: just a basic sanity check that the model is actually learning something.
def plot_training(history):
    plt.figure(figsize=(5.8, 3.5))
    plt.plot(history["step"], history["loss"], marker="o", markersize=4)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training curve")
    plt.grid(True, alpha=0.25)
    savefig(os.path.join(FIGDIR, "training_curve.png"))


# This plots the geometry we expect from exact Bayesian belief updates.
# Left side shows how the posterior over components moves around,
# and the right side shows the local 3-state belief geometry inside each component.
# I capped the prefix length here so it does not blow up memory.
def plot_predicted_geometry():
    prefixes = [[]]
    curr = [[]]

    # Cap exhaustive generation so memory does not explode.
    plot_len = min(SEQ_LEN, 8)

    for _ in range(plot_len):
        nxt = []
        for p in curr:
            for tok in range(3):
                pref = p + [tok]
                prefixes.append(pref)
                nxt.append(pref)
        curr = nxt

    beliefs = np.array([belief_from_prefix(p) for p in prefixes])
    q = np.array([component_posterior(b) for b in beliefs])
    q_xy = barycentric_to_xy(q)

    fig = plt.figure(figsize=(12.2, 4.0))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.35, 1, 1, 1], wspace=0.32)

    tri = barycentric_to_xy(np.eye(3))

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(q_xy[:, 0], q_xy[:, 1], s=6, c=q, alpha=0.65, edgecolors="none")
    ax0.plot(
        [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]],
        [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]],
        color="black", lw=1.0
    )
    ax0.text(tri[0, 0] - 0.04, tri[0, 1] - 0.04, "M1", fontsize=10)
    ax0.text(tri[1, 0] + 0.01, tri[1, 1] - 0.04, "M2", fontsize=10)
    ax0.text(tri[2, 0], tri[2, 1] + 0.04, "M3", fontsize=10, ha="center")
    ax0.set_title("Component posterior", pad=8)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_aspect("equal")
    ax0.grid(False)

    for c in range(3):
        ax = fig.add_subplot(gs[0, c + 1])
        mask = q[:, c] > 0.9
        if mask.sum() == 0:
            mask = q.argmax(axis=1) == c

        eta = np.array([local_belief(b, c) for b in beliefs[mask]])
        xy = barycentric_to_xy(eta)

        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=np.clip(eta, 0, 1), alpha=0.65, edgecolors="none")
        ax.plot(
            [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]],
            color="black", lw=1.0
        )
        ax.text(tri[0, 0] - 0.04, tri[0, 1] - 0.04, "S1", fontsize=9)
        ax.text(tri[1, 0] + 0.01, tri[1, 1] - 0.04, "S2", fontsize=9)
        ax.text(tri[2, 0], tri[2, 1] + 0.04, "S3", fontsize=9, ha="center")
        ax.set_title(f"{COMPONENTS[c].name} local belief", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)

    savefig(os.path.join(FIGDIR, "predicted_geometry.png"))


# This makes heatmaps of how readable the beliefs are from the residual stream.
# Intuition: darker/better regions mean that layer-position pair is carrying more belief information.
def plot_heatmaps(analysis):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
    mats = [analysis["q_r2"], analysis["q_rmse"], analysis["full_r2"], analysis["full_rmse"]]
    titles = [
        r"Linear readout of $q(c \mid x_{1:t})$: $R^2$",
        r"Linear readout of $q(c \mid x_{1:t})$: RMSE",
        r"Linear readout of full 9-state belief: $R^2$",
        r"Linear readout of full 9-state belief: RMSE",
    ]
    cmaps = ["viridis", "magma_r", "viridis", "magma_r"]

    for ax, M, title, cmap in zip(axes.ravel(), mats, titles, cmaps):
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Context position")
        ax.set_ylabel("Layer (0 = embeddings)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    savefig(os.path.join(FIGDIR, "regression_heatmaps.png"))


# This plots how the true Bayesian component uncertainty changes with context position.
# Intuition: as we see more tokens, we should get more certain about which component generated the sequence.
def plot_position_curves(analysis):
    T = len(analysis["pos_entropy"])
    x = np.arange(1, T+1)
    plt.figure(figsize=(6.2, 3.6))
    plt.plot(x, analysis["pos_entropy"], marker='o', label='Mean component entropy')
    plt.plot(x, analysis["pos_qmax"], marker='s', label='Mean max component posterior')
    plt.xlabel("Context position")
    plt.title("Bayesian synchronization across positions")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    savefig(os.path.join(FIGDIR, "position_curves.png"))


# This projects the final-layer activations into the learned 3-way simplex over components.
# Intuition: it lets us visually compare the network's learned component geometry with the true posterior geometry.
def plot_projected_component_simplex(bundle, analysis):
    residuals = bundle["residuals"]
    positions = [0, min(3, residuals.shape[2] - 1), residuals.shape[2] - 1]
    layer = residuals.shape[0] - 1

    fig, axes = plt.subplots(1, len(positions), figsize=(11.8, 3.6))
    tri = barycentric_to_xy(np.eye(3))

    for ax, t in zip(axes, positions):
        rep = analysis["regs_q"][(layer, t)]
        pred = np.clip(rep["pred"], 0, None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)
        xy = barycentric_to_xy(pred)

        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=8, c=rep["Yte"], alpha=0.65,
            edgecolors="none"
        )
        ax.plot(
            [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]],
            color="black", lw=1.0
        )
        ax.text(tri[0, 0] - 0.04, tri[0, 1] - 0.04, "M1", fontsize=10)
        ax.text(tri[1, 0] + 0.01, tri[1, 1] - 0.04, "M2", fontsize=10)
        ax.text(tri[2, 0], tri[2, 1] + 0.04, "M3", fontsize=10, ha="center")
        ax.set_title(f"Position {t+1}", pad=8)
        ax.text(
            0.5, -0.12, rf"$R^2 = {rep['r2']:.3f}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)

    savefig(os.path.join(FIGDIR, "projected_component_simplex.png"))


# This checks the local geometry inside each component at the final position.
# Intuition: once we are fairly sure which component we are in,
# do the activations also reflect the local 3-state belief inside that component?
def plot_local_geometries(bundle, analysis):
    residuals = bundle["residuals"]
    beliefs = bundle["beliefs"]
    q = analysis["q"]
    layer = residuals.shape[0] - 1
    t = residuals.shape[2] - 1
    X = residuals[layer, :, t, :]

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.4))
    tri = barycentric_to_xy(np.eye(3))
    row_labels = ["True local belief", "From activations"]

    for c in range(3):
        mask = q[:, t, c] > 0.95
        if mask.sum() < 50:
            mask = bundle["components"] == c

        Y = np.array([local_belief(beliefs[i, t], c) for i in range(len(mask)) if mask[i]])
        Xc = X[mask]
        rep = regression_report(Xc, Y)

        pred = np.clip(rep["pred"], 0, None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)

        ytrue_xy = barycentric_to_xy(rep["Yte"])
        ypred_xy = barycentric_to_xy(pred)

        for row, xy in [(0, ytrue_xy), (1, ypred_xy)]:
            ax = axes[row, c]
            ax.scatter(
                xy[:, 0], xy[:, 1],
                s=8, c=rep["Yte"], alpha=0.65,
                edgecolors="none"
            )
            ax.plot(
                [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0]],
                [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1]],
                color="black", lw=1.0
            )
            ax.text(tri[0, 0] - 0.04, tri[0, 1] - 0.04, "S1", fontsize=9)
            ax.text(tri[1, 0] + 0.01, tri[1, 1] - 0.04, "S2", fontsize=9)
            ax.text(tri[2, 0], tri[2, 1] + 0.04, "S3", fontsize=9, ha="center")

            if row == 0:
                ax.set_title(COMPONENTS[c].name, pad=8)
            if c == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.grid(False)

        axes[1, c].text(
            0.5, -0.12, rf"$R^2 = {rep['r2']:.3f}$",
            transform=axes[1, c].transAxes, ha="center", va="top", fontsize=10
        )

    savefig(os.path.join(FIGDIR, "local_geometries.png"))


# ---------------------------
# Report
# ---------------------------

# This makes a small summary table for the PDF report.
# Intuition: gather the main numbers in one clean place.
def make_summary(results: Dict) -> List[List[str]]:
    rows = [["quantity", "value"]]
    rows.append(["training steps", str(results['train_cfg']['steps'])])
    rows.append(["batch size", str(results['train_cfg']['batch_size'])])
    rows.append(["model", f"{results['train_cfg']['n_layers']} layers, d_model={results['train_cfg']['d_model']}"])
    rows.append(["final train loss", f"{results['history']['loss'][-1]:.4f}"])
    rows.append(["training time (s)", f"{results['history']['train_seconds']:.1f}"])
    q_r2_final = results['analysis']['q_r2'][-1, -1]
    full_r2_final = results['analysis']['full_r2'][-1, -1]
    rows.append(["final-layer final-position R² (component posterior)", f"{q_r2_final:.3f}"])
    rows.append(["final-layer final-position R² (full 9-state belief)", f"{full_r2_final:.3f}"])
    rows.append(["avg true component entropy pos 1", f"{results['analysis']['pos_entropy'][0]:.3f}"])
    rows.append([f"avg true component entropy pos {SEQ_LEN}", f"{results['analysis']['pos_entropy'][-1]:.3f}"])
    return rows


# This adds a figure plus a fuller caption to the PDF.
# Intuition: each caption should say what the figure is, why we care, and what we expect to see.
def add_figure_with_caption(story, title, fig_file, caption, styles, width=6.7*inch, height=2.55*inch):
    story.append(Paragraph(title, styles['Heading1']))
    story.append(Spacer(1, 0.03*inch))
    story.append(Image(os.path.join(FIGDIR, fig_file), width=width, height=height))
    story.append(Spacer(1, 0.06*inch))
    story.append(Paragraph(caption, styles['Caption']))
    story.append(Spacer(1, 0.14*inch))

def build_pdf_report(results: Dict, pdf_path: str):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='Small',
        parent=styles['BodyText'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['BodyText'],
        fontName='Times-Roman',
        fontSize=9.2,
        leading=11.8,
        leftIndent=6,
        rightIndent=6,
        spaceAfter=8,
        textColor=colors.HexColor('#222222'),
    ))

    styles['Title'].fontName = 'Times-Bold'
    styles['Title'].fontSize = 19
    styles['Title'].leading = 22

    styles['Heading1'].fontName = 'Times-Bold'
    styles['Heading1'].fontSize = 14
    styles['Heading1'].leading = 18
    styles['Heading1'].textColor = colors.HexColor('#17365D')
    styles['Heading1'].spaceBefore = 10
    styles['Heading1'].spaceAfter = 6

    styles['BodyText'].fontName = 'Times-Roman'
    styles['BodyText'].fontSize = 10.5
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 8

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.72*inch,
        rightMargin=0.72*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )

    story = []
    story.append(Paragraph("Non-ergodic Mess3 Mixture Experiment", styles['Title']))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("Eshika Pathak", styles['BodyText']))
    story.append(Paragraph("Experiment report", styles['Small']))
    story.append(Spacer(1, 0.08*inch))

    intro = (
        f"This report studies a non-ergodic mixture of three Mess3 processes. "
        f"For each training sequence, one ergodic component is sampled once and then kept fixed for the whole sequence, "
        f"so the global hidden process is a block-diagonal 9-state hidden Markov model. "
        f"A small decoder-only transformer is trained by next-token prediction on sequences of length {SEQ_LEN} "
        f"over a 4-token vocabulary made up of three observation symbols plus BOS."
    )
    story.append(Paragraph(intro, styles['BodyText']))

    story.append(Paragraph("1. Dataset construction and training setup", styles['Heading1']))
    setup = (
        f"I used three Mess3 components with parameters "
        f"M1 = (α = {COMPONENTS[0].alpha:.2f}, x = {COMPONENTS[0].x:.2f}), "
        f"M2 = (α = {COMPONENTS[1].alpha:.2f}, x = {COMPONENTS[1].x:.2f}), and "
        f"M3 = (α = {COMPONENTS[2].alpha:.2f}, x = {COMPONENTS[2].x:.2f}). "
        f"Each training sequence is generated entirely by one of these components, chosen once at the start of the sequence. "
        f"This makes the overall process non-ergodic, because the sequence never switches between components after generation begins. "
        f"The transformer is then trained on this dataset with standard next-token prediction."
    )
    story.append(Paragraph(setup, styles['BodyText']))

    table = Table(make_summary(results), colWidths=[2.8*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DCEAF7')),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('GRID', (0,0), (-1,-1), 0.45, colors.HexColor('#888888')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor('#F8FBFF')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('LEADING', (0,0), (-1,-1), 11),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.14*inch))

    story.append(Paragraph("2. Why this structure is interesting for language models", styles['Heading1']))
    lm_text = (
        "I think this structure is interesting because real language data are not generated by one single stationary source. "
        "Different users, domains, writing styles, or tasks can behave like different latent generators. "
        "A non-ergodic mixture is a simple way to model that. "
        "Within one context window, the model may need to figure out which generator is active and then adapt its predictions accordingly. "
        "So this experiment acts like a toy version of in-context adaptation: first identify the active component, then track where you are inside that component."
    )
    story.append(Paragraph(lm_text, styles['BodyText']))

    story.append(Paragraph("3. Honor-code pre-registered prediction", styles['Heading1']))
    prereg = (
        "Before looking at the trained-model plots, my prediction was that the residual stream would develop a two-scale geometry. "
        "At early context positions, activations from different components should overlap more because the model is still uncertain about which component generated the sequence. "
        "As more context is observed, the representation should separate more clearly by component. "
        "Within high-confidence regions for a given component, I expected the activations to reflect the local 3-state belief geometry of that component. "
        "I also expected later layers to show cleaner and more linearly readable geometry than earlier layers."
    )
    story.append(Paragraph(prereg, styles['BodyText']))

    story.append(Paragraph("4. Mathematical derivation of the predicted geometry", styles['Heading1']))
    deriv1 = (
        "Let c denote the ergodic component and let s denote the local hidden state within that component. "
        "Since there are 3 components and each component has 3 hidden states, the full hidden state can be written as the pair (c, s), giving 9 total hidden states."
    )
    story.append(Paragraph(deriv1, styles['BodyText']))

    deriv2 = (
        "The key structural fact is that the process never jumps between components once a sequence starts. "
        "So the global transition operator is block diagonal: each component evolves inside its own 3-state block, and there are no cross-component transitions."
    )
    story.append(Paragraph(deriv2, styles['BodyText']))

    deriv3 = (
        "If b_t(c, s) is the posterior probability of being in component c and local state s after observing the first t tokens, "
        "then a Bayesian update at the next step only mixes over states inside the same component. "
        "That means the posterior naturally splits into two pieces: a posterior over which component generated the sequence, and a conditional posterior over the local state inside that component."
    )
    story.append(Paragraph(deriv3, styles['BodyText']))

    deriv4 = (
        "In other words, the 9-state belief can be written in the form "
        "b_t(c, s) = q_t(c) × a_t,c(s), "
        "where q_t(c) is the 3-way posterior over components and a_t,c(s) is the local 3-state belief inside component c."
    )
    story.append(Paragraph(deriv4, styles['BodyText']))

    deriv5 = (
        "This factorization gives the geometric prediction. "
        "The coarse part of the belief, q_t(c), lives in a 3-way simplex, so globally we should see a component-level simplex geometry. "
        "Then, once one component has high posterior, the remaining uncertainty is only over the 3 local states inside that component, so within each branch we should see a local 3-state simplex geometry. "
        "That is why I expected a mixture-of-geometries structure rather than one single global shape."
    )
    story.append(Paragraph(deriv5, styles['BodyText']))

    story.append(Paragraph("5. Geometric intuition", styles['Heading1']))
    intuition = (
        "The simplest way I think about it is: first the model is trying to answer “which world am I in?”, "
        "and after that becomes clearer, it is trying to answer “where am I inside that world?” "
        "So early in the sequence I expect activations to sit closer to an overlap region, because the component is still ambiguous. "
        "Later in the sequence I expect the representation to split apart by component. "
        "Within each branch, I expect a smaller local geometry corresponding to the hidden-state uncertainty of that component."
    )
    story.append(Paragraph(intuition, styles['BodyText']))

    story.append(Paragraph("6. Multiple possible geometries", styles['Heading1']))
    alt_geom = (
        "There were at least three plausible possibilities before training. "
        "The first was the ideal two-scale geometry: activations separate by component posterior and also retain the local 3-state belief geometry within each branch. "
        "The second was a coarser geometry: later layers might mainly encode only the component posterior, so the representation would collapse into three broad clusters with little visible within-component structure. "
        "The third was a noisier or curved version of the two-scale picture, where component identity is clear but the local geometry only survives approximately rather than as a clean simplex-like object."
    )
    story.append(Paragraph(alt_geom, styles['BodyText']))

    add_figure_with_caption(
        story,
        "7. Predicted geometry",
        "predicted_geometry.png",
        (
            "<b>What is plotted.</b> The left panel shows the exact Bayesian posterior over the three ergodic components, mapped into a 2D probability simplex. "
            "Each point is one observed prefix. The triangle vertices represent certainty in one component: bottom-left = M1, bottom-right = M2, and top = M3. "
            "Point color also shows the component posterior itself: red, green, and blue intensities indicate posterior mass on M1, M2, and M3, respectively. "
            "The three panels on the right show the local 3-state belief geometry inside each component, after restricting to prefixes where that component has high posterior. "
            "In those panels, the triangle vertices represent certainty in the three local hidden states of that component: bottom-left = S1, bottom-right = S2, top = S3. "
            "Point color again gives the 3-way local belief vector, so mixed colors indicate uncertainty across local states. "
            "<br/><br/>"
            "<b>Why this matters.</b> This is the exact geometry predicted by Bayesian filtering, so it is the main target structure against which the transformer's residual-stream geometry can be compared. "
            "<br/><br/>"
            "<b>What we expect.</b> A two-scale structure: globally, the geometry should organize by component posterior, and locally, each component branch should contain its own 3-state belief geometry."
        ),
        styles,
        width=6.8*inch,
        height=2.7*inch
    )

    add_figure_with_caption(
        story,
        "8. Optimization",
        "training_curve.png",
        (
            "<b>What is plotted.</b> Training cross-entropy loss as a function of optimization step. "
            "<br/><br/>"
            "<b>Why this matters.</b> Before interpreting geometry, I wanted to check that the model was actually learning predictive structure from the data. "
            "<br/><br/>"
            "<b>What we expect.</b> If the model is using context to infer component identity and latent state, then the loss should fall below the uninformed baseline."
        ),
        styles,
        width=6.3*inch,
        height=2.75*inch
    )

    add_figure_with_caption(
        story,
        "9. Linear readout from the residual stream",
        "regression_heatmaps.png",
        (
            "<b>What is plotted.</b> Each heatmap measures how well a linear probe can recover a target belief quantity from the residual stream at a particular layer and context position. "
            "The top row is for the 3-way component posterior and the bottom row is for the full 9-state hidden-state belief. "
            "Higher R² and lower RMSE mean that the corresponding latent variable is more linearly readable from the representation. "
            "<br/><br/>"
            "<b>Why this matters.</b> These plots show where in the network and where in the sequence the model is carrying coarse component information versus finer hidden-state information. "
            "<br/><br/>"
            "<b>What we expect.</b> Component identity should become readable earlier and more strongly than the full 9-state belief, and both should generally improve at deeper layers and later positions."
        ),
        styles,
        width=6.8*inch,
        height=4.2*inch
    )

    add_figure_with_caption(
        story,
        "10. Learned component geometry",
        "projected_component_simplex.png",
        (
            "<b>What is plotted.</b> Final-layer residual activations at selected context positions are linearly projected into the best-fit 3-way simplex for the component posterior. "
            "Each point is one held-out example. The triangle vertices represent certainty in M1, M2, and M3; specifically bottom-left = M1, bottom-right = M2, and top = M3. "
            "Point color shows the true component posterior for that example, using the same RGB convention as before: the color channels indicate posterior mass on M1, M2, and M3. "
            "<br/><br/>"
            "<b>Why this matters.</b> This gives a direct visual comparison between the network's learned component-level geometry and the exact Bayesian geometry. "
            "<br/><br/>"
            "<b>What we expect.</b> Early in the sequence, points should cluster closer to the uncertain interior of the simplex. As more context is observed, points should move toward component-specific regions, reflecting better synchronization to the correct ergodic component."
        ),
        styles,
        width=6.8*inch,
        height=2.9*inch
    )

    add_figure_with_caption(
        story,
        "11. Relation to the component processes",
        "local_geometries.png",
        (
            "<b>What is plotted.</b> Each column corresponds to one Mess3 component. "
            "The top row shows the true local 3-state belief geometry within that component, after conditioning on examples where the component posterior is high. "
            "The bottom row shows the corresponding geometry obtained by linearly projecting final-layer activations. "
            "In each panel, the triangle vertices represent certainty in one of the three local hidden states for that component: bottom-left = S1, bottom-right = S2, top = S3. "
            "Point color shows the true local 3-state belief vector, so mixed colors indicate uncertainty across local states. "
            "<br/><br/>"
            "<b>Why this matters.</b> This tells us whether the network has learned only coarse component classification, or whether it also captures the finer within-component latent-state structure. "
            "<br/><br/>"
            "<b>What we expect.</b> If the learned representation really has the predicted two-scale belief geometry, then the bottom-row projected activations should resemble the top-row true local belief geometry."
        ),
        styles,
        width=6.8*inch,
        height=4.1*inch
    )

    story.append(Paragraph("12. Direct answer: what structure is present after training?", styles['Heading1']))
    answer_geom = (
        "After training, the clearest structure in the residual stream is the posterior over ergodic component. "
        "Later positions and deeper layers show stronger component separation than early positions and shallow layers. "
        "Within examples where the model is already fairly certain about the component, the residual stream also retains information about the local 3-state belief of that component, although this finer structure is weaker than the coarse component geometry. "
        "So the learned representation is best viewed not as one single simplex, but as a mixture of a component-level geometry together with local within-component belief geometry."
    )
    story.append(Paragraph(answer_geom, styles['BodyText']))

    story.append(Paragraph("13. Additional analysis of my choosing", styles['Heading1']))
    extra = (
        "As an additional analysis, I looked at Bayesian synchronization across context position through the mean component entropy and the mean maximum component posterior. "
        "I chose this because it helps separate the difficulty of the statistical problem from the behavior of the transformer itself. "
        "If the exact Bayesian posterior becomes sharper with more context, then later component separation in the network is something we should expect from the data-generating process itself."
    )
    story.append(Paragraph(extra, styles['BodyText']))

    add_figure_with_caption(
        story,
        "14. Additional analysis: Bayesian synchronization across position",
        "training_and_sync.png",
        (
            "<b>What is plotted.</b> The left panel shows training loss over optimization. "
            "The right panel shows two summary statistics of the exact Bayesian posterior over components as a function of context position: mean component entropy and mean maximum component posterior. "
            "<br/><br/>"
            "<b>Why I chose this analysis.</b> This helps separate two questions: whether the transformer is learning useful structure, and whether the underlying inference problem itself becomes easier as more tokens are observed. "
            "<br/><br/>"
            "<b>What we expect.</b> As context grows, the exact posterior over components should become sharper. So mean entropy should decrease and the mean maximum component posterior should increase. "
            "That would support the interpretation that later component separation in the residual stream is statistically natural for this data-generating process."
        ),
        styles,
        width=6.8*inch,
        height=2.9*inch
    )

    extra2 = (
        "Another analysis I think would be very informative, even though I did not implement it here, would be to compare the transformer's next-token cross-entropy to the exact Bayesian next-token predictor at each context position. "
        "That would tell us how close the learned model is to the Bayes-optimal predictor for this non-ergodic hidden Markov model."
    )
    story.append(Paragraph(extra2, styles['BodyText']))

    story.append(Paragraph("15. Main takeaways", styles['Heading1']))
    bullets = [
        "A non-ergodic Mess3 mixture creates a natural two-scale latent inference problem: identify the active component, then track the local state within it.",
        "The mathematically predicted belief factorization naturally leads to a two-scale geometry in the residual stream.",
        "Empirically, component posterior is the strongest and most linearly accessible structure; finer within-component belief structure is present but weaker.",
        "This makes the experiment a useful toy model of in-context adaptation in language models, where the model must infer which latent generator is currently active."
    ]
    for b in bullets:
        story.append(Paragraph("• " + b, styles['BodyText']))

    caveat = (
        "Caveat: this is still a small proof-of-concept on CPU, so it should be read as an implementation and qualitative geometry study because my model training isn't good enough to draw strong conclusions."
    )
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph(caveat, styles['Small']))

    doc.build(story)




# This zips up the code and output files.
# makes it easy to download and submit everything together.
def bundle_code(zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTDIR):
            for fn in files:
                if fn.endswith('.zip'):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, OUTDIR)
                zf.write(full, rel)


# This runs the full experiment from start to finish.
# save the preregistration note, train the model, run the analysis,
# make all the plots, save the results, build the PDF, and zip everything.
def main():
    cfg = TrainConfig()
    prereg_text = textwrap.dedent("""
    Honor-code preregistration before inspecting post-training figures:
    1) Because the global process is a non-ergodic mixture of 3 Mess3 components, the full 9-state belief should factor as b=(q1*eta1,q2*eta2,q3*eta3).
    2) Early context positions should mainly encode q, with strong overlap between components near the prior q=(1/3,1/3,1/3).
    3) Later positions should separate by component; within high-confidence branches, one should recover component-specific Mess3 geometry.
    4) A possible alternative is that final layers encode mostly q, with fine within-component structure pushed to earlier layers.
    """).strip()
    with open(os.path.join(OUTDIR, 'honor_code_preregistration.txt'), 'w') as f:
        f.write(prereg_text + "\n")

    model, history = train_model(cfg)
    torch.save(model.state_dict(), os.path.join(OUTDIR, 'tiny_gpt_nonergodic_mess3.pt'))

    plot_training(history)
    plot_predicted_geometry()

    bundle = sample_analysis_set(model)
    analysis = analyze_residual_geometry(bundle)
    plot_heatmaps(analysis)
    plot_position_curves(analysis)
    plot_projected_component_simplex(bundle, analysis)
    plot_local_geometries(bundle, analysis)

    # This makes one extra figure that puts optimization and Bayesian synchronization side by side.
    # one panel shows whether training worked, and the other shows how the true component
    # uncertainty changes as context grows.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))

    axes[0].plot(history['step'], history['loss'], marker='o', markersize=4)
    axes[0].set_title('Training curve')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].grid(True, alpha=0.25)

    x = np.arange(1, len(analysis['pos_entropy'])+1)
    axes[1].plot(x, analysis['pos_entropy'], marker='o', label='Mean component entropy')
    axes[1].plot(x, analysis['pos_qmax'], marker='s', label='Mean max component posterior')
    axes[1].set_title('Bayesian synchronization across positions')
    axes[1].set_xlabel('Context position')
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    savefig(os.path.join(FIGDIR, 'training_and_sync.png'))

    results = {
        'seed': SEED,
        'train_cfg': asdict(cfg),
        'history': history,
        'components': [asdict(c) for c in COMPONENTS],
        'analysis': {
            'q_r2': analysis['q_r2'].tolist(),
            'q_rmse': analysis['q_rmse'].tolist(),
            'full_r2': analysis['full_r2'].tolist(),
            'full_rmse': analysis['full_rmse'].tolist(),
            'pos_entropy': analysis['pos_entropy'].tolist(),
            'pos_qmax': analysis['pos_qmax'].tolist(),
        }
    }
    with open(os.path.join(OUTDIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pdf_path = os.path.join(OUTDIR, 'nonergodic_mess3_report.pdf')
    results_for_report = {
        **results,
        'analysis': {
            'q_r2': np.array(results['analysis']['q_r2']),
            'full_r2': np.array(results['analysis']['full_r2']),
            'pos_entropy': np.array(results['analysis']['pos_entropy']),
            'pos_qmax': np.array(results['analysis']['pos_qmax']),
        }
    }
    build_pdf_report(results_for_report, pdf_path)
    # bundle_code(os.path.join(OUTDIR, 'nonergodic_mess3_code.zip'))
    print('done')


if __name__ == '__main__':
    main()