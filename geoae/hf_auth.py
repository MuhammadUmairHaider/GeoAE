"""
Hugging Face authentication for gated checkpoints (Gemma 3, Llama, …).

`google/gemma-3-12b-pt` is gated: without a token every `from_pretrained` call
fails with a 401 that looks like a network error. This module resolves a token
once per process and exports it as `HF_TOKEN`, which is what huggingface_hub
reads — so transformers, datasets and the hub client all pick it up.

The token is never written into source. Resolution order:

    1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN already in the environment
    2. GEOAE_HF_TOKEN_FILE=<path>
    3. a token file in the package root, repo root, or cwd
       (hf_tokken.txt, hf_token.txt, .hf_token — all git-ignored)
    4. an existing `hf auth login` credential in ~/.cache/huggingface

Missing tokens are NOT fatal: public models keep working, and the gated ones
raise `load_lm`'s actionable GATED_HINT instead of a bare 401.
"""
from __future__ import annotations

import os
from pathlib import Path

from geoae.paths import PACKAGE_ROOT, REPO_ROOT

TOKEN_FILENAMES = ("hf_tokken.txt", "hf_token.txt", ".hf_token")
_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_resolved: str | None = None   # cached source label; set once login succeeds


def _candidate_files() -> list[Path]:
    override = os.environ.get("GEOAE_HF_TOKEN_FILE")
    files = [Path(override)] if override else []
    seen: set[Path] = set()
    for base in (PACKAGE_ROOT, REPO_ROOT, Path.cwd()):
        for name in TOKEN_FILENAMES:
            p = base / name
            if p not in seen:
                seen.add(p)
                files.append(p)
    return files


def find_token() -> tuple[str | None, str]:
    """Return (token, human-readable source). Never logs the token itself."""
    for key in _ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val, f"${key}"

    for path in _candidate_files():
        try:
            if path.is_file():
                val = path.read_text().strip()
                if val:
                    return val, str(path)
        except OSError:
            continue

    try:
        from huggingface_hub import get_token
        val = (get_token() or "").strip()
        if val:
            return val, "hf auth login cache"
    except Exception:
        pass

    return None, "not found"


def ensure_hf_login(verbose: bool = True) -> bool:
    """
    Make a token visible to every HF library in this process. Idempotent and
    offline (no whoami round-trip). Returns True if a token is now in place.
    """
    global _resolved
    if _resolved is not None:
        return True

    token, source = find_token()
    if token is None:
        if verbose:
            print("[hf_auth] no HF token found — gated repos (Gemma, Llama) will 401. "
                  f"Put one in {PACKAGE_ROOT / TOKEN_FILENAMES[0]} or run `hf auth login`.")
        return False

    for key in _ENV_KEYS:
        os.environ[key] = token
    _resolved = source
    if verbose:
        print(f"[hf_auth] HF token loaded from {source} (…{token[-4:]})")
    return True


def whoami() -> str | None:
    """Username for the resolved token, or None. Makes a network call."""
    if not ensure_hf_login(verbose=False):
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi(token=os.environ["HF_TOKEN"]).whoami().get("name")
    except Exception:
        return None
