"""All hyperparameters in one place. Override via configs/default.yaml or a custom YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class ExtractionConfig:
    model_name: str = "meta-llama/Llama-3.2-3B"
    layers: List[int] = field(default_factory=lambda: [16, 20, 24, 27])
    target_layer: int = 27
    hidden_size: int = 3072
    pack_size: int = 16384        # (legacy, unused by per-doc extractor)
    skip_first_n: int = 8         # (legacy, unused by per-doc extractor)
    max_doc_tokens: int = 256     # truncate each doc (balances domains, diversifies contexts)
    skip_leading: int = 4         # skip first N tokens of each doc (BOS / doc-start flood)
    n_tokens: int = 5_000_000
    dtype: str = "float16"        # storage dtype for .npy files
    activations_dir: str = "activations"
    log_every: int = 10           # batches between progress prints

    # Data sources (easily swappable)
    data_sources: List[str] = field(default_factory=lambda: [
        "allenai/c4",
        "wikipedia",
        "bookcorpus",
        "codeparrot/github-code",
        "cc_news",
        "togethercomputer/RedPajama-Data-1T-Sample",
    ])


@dataclass
class DataConfig:
    activations_dir: str = "activations"
    target_layer: int = 27
    batch_size: int = 4096
    val_frac: float = 0.05


@dataclass
class ModelConfig:
    hidden_size: int = 3072
    latent_dim: int = 2048
    n_clusters: int = 128
    nonlinearity: str = "linear"   # "linear" | "relu" | "gelu"
    metric: str = "euclidean"      # "euclidean" (magnitude) | "cosine" (directional)
    latent_norm: str = "none"      # "none" | "batch" (BatchNorm1d between linear and act)


@dataclass
class LossConfig:
    lambda_cluster: float = 0.1
    lambda_sep: float = 0.01
    sep_mode: str = "median"       # "median" (legacy, scale-free) | "intra" | "hinge"
    sep_margin: float = 2.0        # hinge only: target d_min >= margin * intra radius
    lambda_usage: float = 0.1
    lambda_mse: float = 0.0   # weak MSE recon anchor for e2e KL training (0 = off)
    lambda_var: float = 0.0   # VICReg variance hinge on latents (anti-contraction, 0 = off)
    lambda_cov: float = 0.0   # VICReg covariance on latents (anti low-rank collapse, 0 = off)
    lambda_unif: float = 0.0  # Wang–Isola uniformity on unit-sphere latents (cosine runs, 0 = off)
    sinkhorn_iters: int = 3
    # --- generalised cluster-marginal balancing (all defaults = today's behaviour) ---
    balance: str = "uniform"       # "uniform" | "zipf"  — shape of the column target
    balance_rho: float = 1.0       # column-constraint strength: 1 = hard Sinkhorn, 0 = softmax
    balance_eta: float = 1.0       # cross-batch EMA on the column dual: 1 = per-batch
    zipf_alpha_start: float = 0.0  # annealed 0 -> zipf_alpha_end over the tau window
    zipf_alpha_end: float = 0.0
    tau_start: float = 1.0
    tau_end: float = 0.1


@dataclass
class TrainConfig:
    n_epochs: int = 15
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    ema_hard: bool = False   # centroid EMA from hard (argmax) assignments instead of soft Q

    # Training schedule phase boundaries (inclusive epoch numbers, 1-indexed)
    recon_only_epochs: int = 2       # epochs 1..recon_only_epochs: pure reconstruction
    clustering_start_epoch: int = 3  # epoch where clustering loss is added
    full_loss_start_epoch: int = 6   # epoch where sep+usage losses are added
    geometry_start_epoch: int = 0    # epoch where var/cov/unif switch on.
                                     # 0 = on from step 1 (previous behaviour, unchanged).
                                     # Set >1 for an explicit recon-only warmup before VICReg.

    diag_every: int = 100            # steps between diagnostics
    reinit_every: int = 1000         # steps between dead-cluster reinitialization
    save_every: int = 1              # save checkpoint every N epochs
    keep_checkpoints: int = 3        # keep last N + best-by-val-recon

    centroid_init: str = "kmeans++"   # "kmeans++" | "semisup" | "class_means"
    semisup_cap: int = 500            # semisup init: max labeled samples/class for class-mean centroids
    teacher_mode: str = "cached"      # e2e only: "cached" (precomputed logits) |
                                      # "onfly" (teacher = head(norm(x)) in-loop, no cache)
    checkpoints_dir: str = "checkpoints"
    runs_dir: str = "runs"
    seed: int = 42

    wandb_project: str = "geoae"
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = cls()
        for section, values in overrides.items():
            sub = getattr(cfg, section, None)
            if sub is None:
                raise ValueError(f"Unknown config section: {section!r}")
            for k, v in values.items():
                if not hasattr(sub, k):
                    raise ValueError(f"Unknown config key: {section}.{k!r}")
                setattr(sub, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
