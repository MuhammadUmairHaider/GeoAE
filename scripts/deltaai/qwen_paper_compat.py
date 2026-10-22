"""Small nonlinear-decoder compatibility hunks for the newer canonical evaluators.

Keep the newer paper code, including its earlier-layer support. Only restore the
current implementation's exact decoded changes and explicit fixed-direction guards.
Every old text must occur exactly once; validate all files before writing any.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


PATCHES = {
    "geoae/interp/concept_remove_generate.py": (
        ('            x2 = x + g[:, None] * ae.decoder(edit_fn(a) - a)',
         '            x2 = x + g[:, None] * ae.decoded_delta(a, edit_fn(a) - a, tok)'),
        ('g*ae.decoder(edit(z) - z). The decoder is linear and bias-free, and the token-bypass table\n'
         'only depends on the token, not on z -- see concept_steer_generate.decode_direction / this\n'
         "file's own tests for the cancellation -- so an IDENTITY edit (or g=0 throughout) leaves",
         'g*[ae.decode(edit(z), tok) - ae.decode(z, tok)]. The token-bypass table depends only\n'
         'on the current token, so it cancels for both linear and nonlinear decoders.\n'
         'An IDENTITY edit (or g=0 throughout) leaves'),
        ('    z (delta)    x\' = x + g*ae.decoder(edit_fn(z) - z)          (identity edit or g=0 -> x\'==x)',
         '    z (delta)    x\' = x + g*[decode(edit_fn(z), tok) - decode(z, tok)]\n'
         '                 (identity edit or g=0 -> x\'==x, including nonlinear decoders)'),
    ),
    "geoae/interp/concept_steer_generate.py": (
        ('    return ae.decoder(v_z)\n',
         '    ae.require_linear_decoder("label_z fixed global steering direction")\n'
         '    return ae.decoder(v_z)\n'),
        ('    mean, std = arms["base"].mean, arms["base"].std\n',
         '    if "label_z" in args.handles:\n'
         '        arms["bypass"].ae.require_linear_decoder("label_z fixed global steering direction")\n'
         '    mean, std = arms["base"].mean, arms["base"].std\n'),
        ('    normalised residual space using the decoder WEIGHT only: u = ae.decoder(v_z).',
         '    normalised residual space using a bias-free linear decoder: u = ae.decoder(v_z).\n\n'
         '    Nonlinear decoder changes require the current latent state; use\n'
         '    ae.decoded_delta(z, delta, tok) rather than a fixed global direction.'),
    ),
    "geoae/interp/logit_attribution.py": (
        ('    ae, norm_mean, norm_std, ckpt = load_ae_checkpoint(ckpt_path, device)\n',
         '    ae, norm_mean, norm_std, ckpt = load_ae_checkpoint(ckpt_path, device)\n'
         '    ae.require_linear_decoder("direct logit attribution with fixed centroid directions")\n'),
        ('    device = W_U.device\n',
         '    # KMeansShim supplies an identity decoder and no GeoAE methods.\n'
         '    GeoAE.require_linear_decoder(ae, "direct logit attribution with fixed centroid directions")\n'
         '    device = W_U.device\n'),
    ),
    "geoae/interp/faithfulness_ablate.py": (
        ('        return self.ae.decoder(v) * self.std\n',
         '        self.ae.require_linear_decoder("faithfulness ablation with fixed concept directions")\n'
         '        return self.ae.decoder(v) * self.std\n'),
        ('    enc = KMeansEnc(args.baseline_kmeans, dev) if args.baseline_kmeans else ref\n',
         '    enc = KMeansEnc(args.baseline_kmeans, dev) if args.baseline_kmeans else ref\n'
         '    if not args.baseline_kmeans:\n'
         '        ref.ae.require_linear_decoder("faithfulness ablation with fixed concept directions")\n'),
    ),
}


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def patch_nonlinear_evaluators(source: Path) -> dict:
    """Apply reviewed tiny hunks to an unpublished source snapshot and return provenance."""
    pending = {}
    provenance = {}
    for name, replacements in PATCHES.items():
        path = source / name
        original = path.read_text()
        updated = original
        for before, after in replacements:
            if updated.count(before) != 1:
                raise ValueError(f"Canonical nonlinear compatibility anchor changed in {name}: {before!r}")
            updated = updated.replace(before, after, 1)
        compile(updated, str(path), "exec")
        pending[path] = updated
        provenance[name] = {"before_sha256": _sha(original), "after_sha256": _sha(updated),
                            "hunks": len(replacements)}
    for path, updated in pending.items():
        path.write_text(updated)
    return provenance
