"""Direct inference for the native Google Gemma Scope 2 residual JumpReLU SAEs.

The published ``params.safetensors`` are already expressed in model activation
units. In particular, neither activation normalization nor subtraction of
``b_dec`` belongs in the encoder. The official Google tutorial implements
``relu(x @ w_enc + b_enc) * (x @ w_enc + b_enc > threshold)`` and decodes with
``features @ w_dec + b_dec``. Decoder rows below retain their original norms.

Primary sources (checked 2026-10-04):
* https://huggingface.co/google/gemma-scope-2-4b-pt/blob/main/
  resid_post_all/layer_33_width_16k_l0_small/config.json
* https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r

Capture ``model.layers.33.output`` using a block forward hook: the last entry of
Hugging Face ``hidden_states`` can include the model's final normalization.
Exclude BOS and padding when evaluating these SAEs; Google did not train on BOS.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from torch import nn

DEFAULT_REPO = "google/gemma-scope-2-4b-pt"
DEFAULT_SAE_PATH = "resid_post_all/layer_33_width_16k_l0_small"
DEFAULT_REVISION = "a0ffd6132a985bc84077a66d1a1033e10b604fa8"
_WEIGHT_KEYS = frozenset({"w_enc", "w_dec", "b_enc", "b_dec", "threshold"})
_RESIDUAL_HOOK = re.compile(r"model\.layers\.(\d+)\.output\Z")


def _validate_config(config: Mapping[str, Any], expected_model: str | None,
                     expected_layer: int | None) -> None:
    """Fail before weight download if this is a different model, site or format."""
    if config.get("architecture") != "jump_relu" or config.get("type") != "sae":
        raise ValueError("Expected a native Gemma Scope 2 JumpReLU SAE, not a transcoder/crosscoder")
    if config.get("affine_connection", False) is not False:
        raise ValueError("Affine skip connections are not supported by residual feature ablation")
    if expected_model is not None and config.get("model_name") != expected_model:
        raise ValueError(f"SAE model {config.get('model_name')!r} does not match {expected_model!r}")
    hook_in, hook_out = config.get("hf_hook_point_in"), config.get("hf_hook_point_out")
    match = _RESIDUAL_HOOK.fullmatch(hook_in) if isinstance(hook_in, str) else None
    if match is None or hook_in != hook_out:
        raise ValueError("Expected matching residual block-output input/output hooks")
    if expected_layer is not None and int(match[1]) != expected_layer:
        raise ValueError(f"SAE hook {hook_in!r} does not match layer {expected_layer}")
    width = config.get("width")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("SAE config requires a positive integer width")

    # Native Scope 2 exports have none of these fields. Accept only explicit
    # identity settings, so an alien config cannot silently change raw units.
    for key in ("apply_b_dec_to_input", "subtract_b_dec_from_input", "center_input"):
        if config.get(key, False) is not False:
            raise ValueError(f"Unsupported encoder centering option {key!r}")
    for key in ("input_scale", "output_scale", "activation_norm_scaling_factor"):
        scale = config.get(key, 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (float, int)) or scale != 1.0:
            raise ValueError(f"Unsupported activation scaling option {key!r}; native weights use raw units")
    for key in ("normalize_activations", "normalization"):
        if config.get(key) not in (None, False, "none", "None"):
            raise ValueError(f"Unsupported activation normalization option {key!r}")


class GemmaScopeSAE(nn.Module):
    """A frozen native JumpReLU checkpoint, with all operations in FP32.

    ``feature_delta(z, ids, alpha)`` is the raw residual contribution to
    subtract from a block output. Keeping the residual's reconstruction error
    and the decoder bias intact implements an error-preserving feature edit.
    """

    def __init__(self, params: Mapping[str, torch.Tensor], config: Mapping[str, Any], *,
                 metadata: Mapping[str, Any] | None = None):
        super().__init__()
        _validate_config(config, expected_model=None, expected_layer=None)
        missing, extra = _WEIGHT_KEYS - params.keys(), params.keys() - _WEIGHT_KEYS
        if missing or extra:
            raise ValueError(f"Unexpected checkpoint keys: missing={sorted(missing)}, extra={sorted(extra)}")
        w_enc = params["w_enc"]
        if w_enc.ndim != 2:
            raise ValueError("w_enc must have shape [d_model, n_features]")
        self.d_model, self.n_features = map(int, w_enc.shape)
        self.width = self.n_features
        if self.d_model < 1 or self.n_features != config["width"]:
            raise ValueError("Encoder dimensions disagree with config width")
        expected_shapes = {
            "w_enc": (self.d_model, self.n_features),
            "w_dec": (self.n_features, self.d_model),
            "b_enc": (self.n_features,),
            "b_dec": (self.d_model,),
            "threshold": (self.n_features,),
        }
        device = w_enc.device
        for name, shape in expected_shapes.items():
            value = params[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
            if not value.is_floating_point() or value.device != device:
                raise ValueError(f"{name} must be floating point on the encoder's device")
            value = value.detach().to(dtype=torch.float32).contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains nonfinite checkpoint values")
            if name == "threshold" and (value < 0).any().item():
                raise ValueError("JumpReLU thresholds must be nonnegative")
            self.register_buffer(name, value)
        self.metadata = dict(metadata or {})
        self.metadata.update({
            "config": dict(config), "d_model": self.d_model, "n_features": self.n_features,
            "model_name": config["model_name"], "hook_point": config["hf_hook_point_in"],
            "l0": config.get("l0"), "activation_units": "raw_model_residual",
            "encoder_centers_on_b_dec": False, "decoder_directions_normalized": False,
        })
        self.eval()

    @property
    def decoder_directions(self) -> torch.Tensor:
        """Native decoder rows [n_features, d_model], in raw residual units."""
        return self.w_dec

    @property
    def parameter_count(self) -> int:
        """Number of pretrained scalar weights, including biases and thresholds."""
        return sum(value.numel() for value in self.buffers())

    def _input(self, value: torch.Tensor, width: int, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f"{label} must be a tensor with last dimension {width}")
        if not value.is_floating_point():
            raise ValueError(f"{label} must be floating point")
        if value.device != self.w_enc.device:
            raise ValueError(f"{label} is on {value.device}, but SAE is on {self.w_enc.device}")
        return value.to(dtype=torch.float32)

    @torch.no_grad()
    def encode(self, h_raw_fp32: torch.Tensor) -> torch.Tensor:
        """Encode raw block outputs; features exactly at threshold stay zero."""
        h = self._input(h_raw_fp32, self.d_model, "raw residual")
        with torch.autocast(device_type=h.device.type, enabled=False):
            pre = h @ self.w_enc + self.b_enc
            return torch.relu(pre) * (pre > self.threshold)

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw block outputs, retaining the native decoder bias."""
        z = self._input(features, self.n_features, "features")
        with torch.autocast(device_type=z.device.type, enabled=False):
            return z @ self.w_dec + self.b_dec

    @torch.no_grad()
    def forward(self, h_raw_fp32: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(h_raw_fp32))

    @torch.no_grad()
    def feature_delta(self, features: torch.Tensor, feature_ids: Sequence[int] | torch.Tensor,
                      alpha: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Return ``alpha * z[..., ids] @ w_dec[ids]`` to subtract from raw h.

        IDs must be unique. A scalar alpha or an alpha tensor broadcasting to
        the leading token/batch dimensions is accepted; 1 is full ablation.
        """
        z = self._input(features, self.n_features, "features")
        ids = torch.as_tensor(feature_ids, device=z.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("feature_ids must be a one-dimensional integer sequence")
        ids = ids.to(dtype=torch.long)
        if ids.numel() and ((ids < 0).any().item() or (ids >= self.n_features).any().item()):
            raise ValueError("feature_ids contains an out-of-range feature")
        if ids.unique().numel() != ids.numel():
            raise ValueError("feature_ids must be unique to avoid subtracting contributions twice")
        if isinstance(alpha, torch.Tensor):
            gain = alpha.to(device=z.device, dtype=torch.float32)
            if not torch.isfinite(gain).all().item():
                raise ValueError("alpha must be finite")
            if gain.ndim == z.ndim and gain.shape[-1] == 1:
                gain = gain.squeeze(-1)
            try:
                gain_shape = torch.broadcast_shapes(gain.shape, z.shape[:-1])
            except RuntimeError as exc:
                raise ValueError("alpha must broadcast to token/batch dimensions") from exc
            if gain_shape != z.shape[:-1]:
                raise ValueError("alpha must not change residual shape")
            if gain.ndim:
                gain = gain.unsqueeze(-1)
        else:
            if not math.isfinite(float(alpha)):
                raise ValueError("alpha must be finite")
            gain = float(alpha)
        with torch.autocast(device_type=z.device.type, enabled=False):
            delta = z.index_select(-1, ids) @ self.w_dec.index_select(0, ids)
            result = delta * gain
        if result.shape != delta.shape:
            raise ValueError("alpha must broadcast to token/batch dimensions without changing residual shape")
        return result


def _snapshot_revision(path: str) -> str | None:
    parts = Path(path).parts
    for i, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[i + 1]
    return None


def load_gemma_scope(repo_id: str = DEFAULT_REPO, sae_path: str = DEFAULT_SAE_PATH, *,
                     revision: str | None = None, device: str | torch.device = "cpu",
                     local_dir: str | Path | None = None,
                     expected_model: str | None = "google/gemma-3-4b-pt",
                     expected_layer: int | None = 33) -> GemmaScopeSAE:
    """Load just config.json and params.safetensors, without SAE Lens or training.

    The default 4B release is pinned to a verified Hub revision. Other releases
    resolve a config snapshot once and use that same revision for weights.
    ``local_dir`` accepts a checkpoint directory containing both files or a
    root containing ``sae_path``. Complete local files require no network.
    Missing local files are downloaded into that root by huggingface_hub.
    Local files' revision is unverified and is labeled accordingly in metadata.
    """
    relative = PurePosixPath(sae_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("sae_path must be a relative Hub subdirectory without '..'")
    requested_revision = revision
    selected_revision = revision or (DEFAULT_REVISION if repo_id == DEFAULT_REPO else None)
    config_path = params_path = None
    source = "huggingface_hub"
    resolved_revision = None
    if local_dir is not None:
        root = Path(local_dir)
        for directory in (root / str(relative), root):
            if (directory / "config.json").is_file() and (directory / "params.safetensors").is_file():
                config_path, params_path = directory / "config.json", directory / "params.safetensors"
                source = "local_files"
                resolved_revision = _snapshot_revision(str(config_path))
                break
    if config_path is None:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(
            repo_id=repo_id, filename=f"{relative}/config.json", revision=selected_revision,
            local_dir=local_dir,
        ))
        resolved_revision = _snapshot_revision(str(config_path))
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config, expected_model, expected_layer)
    if params_path is None:
        # local_dir downloads have no snapshot component. The default and
        # explicitly pinned revisions are safe; resolve other revisions once.
        if resolved_revision is None:
            if selected_revision is not None and re.fullmatch(r"[0-9a-f]{40}", selected_revision):
                resolved_revision = selected_revision
            else:
                from huggingface_hub import HfApi

                resolved_revision = HfApi().model_info(repo_id, revision=selected_revision).sha
            # Re-fetch config from the resolved commit in case the branch moved.
            config_path = Path(hf_hub_download(
                repo_id=repo_id, filename=f"{relative}/config.json", revision=resolved_revision,
                local_dir=local_dir,
            ))
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            _validate_config(config, expected_model, expected_layer)
        params_path = Path(hf_hub_download(
            repo_id=repo_id, filename=f"{relative}/params.safetensors", revision=resolved_revision,
            local_dir=local_dir,
        ))
    from safetensors.torch import load_file

    params = load_file(str(params_path), device="cpu")
    sae = GemmaScopeSAE(params, config, metadata={
        "repo_id": repo_id, "sae_path": str(relative), "requested_revision": requested_revision,
        "revision": resolved_revision,
        "revision_verified": source == "huggingface_hub" and resolved_revision is not None,
        "source": source, "config_path": str(config_path), "params_path": str(params_path),
        "source_code": "https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r",
    })
    return sae.to(device=device, dtype=torch.float32).eval()
