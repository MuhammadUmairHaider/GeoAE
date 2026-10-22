# Paper source

`paper.tex` is the authoring copy of the linear-decoder GeoAE paper. Its scientific text and embedded tables remain byte-identical to the recovered version. Overleaf remains the paper working copy; reconcile later authoring changes there deliberately. The [nonlinear-decoder study](../../experiments/nonlinear_decoders/README.md) is separate follow-up work.

The local source is complete for rendering: `custom.bib` came from the frozen evaluation paper source, and `acl.sty` plus `acl_natbib.bst` are unmodified official ACL assets pinned to a commit in [SOURCE.json](SOURCE.json). Upstream notices remain in their headers. This pinned version is not a claim that its styles match the existing Overleaf project byte-for-byte.

The unchanged source built successfully into a [17-page PDF](../../../GeoAE-artifacts/paper_build_20261022/paper.pdf), with all 79 bibliography entries and no undefined citations or references. All pages were rendered and inspected. Two existing overfull-line warnings remain, in an equation and a bibliography entry; their details are in [BUILD.json](BUILD.json). Cleanup preserved the authoring source rather than editing those layouts. This verifies rendering, not regeneration of the scientific tables.

A portable Apple Silicon Tectonic 0.17.0 compiler and its verified upstream download receipt are stored outside the code tree. From the project root, rebuild with the existing cached dependencies:

```bash
mkdir -p ../GeoAE-artifacts/paper_build_20261022
TECTONIC_CACHE_DIR=../GeoAE-artifacts/tools/tectonic-cache \
  ../GeoAE-artifacts/tools/tectonic-0.17.0/tectonic \
  --only-cached --untrusted --keep-logs \
  --outdir ../GeoAE-artifacts/paper_build_20261022 docs/paper/paper.tex
```

On another machine, use the corresponding official Tectonic binary or a TeX distribution with `latexmk`, the packages named by the source, TikZ and Inconsolata. The first Tectonic build can fetch TeX dependencies if `--only-cached` is omitted; it needs no research models or activation dumps. With a complete TeX installation, an alternative from this directory is:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../../GeoAE-artifacts/paper_build_20261022 paper.tex
```

[dependencies.json](dependencies.json) records source hashes, recovered table generators and known missing result inputs. Some result JSONs and K-sweep label arrays were absent from both local research and Git copies. The direct Delta lookup could not authenticate in batch mode, so candidate remote locations remain unverified. Retrieve original artifacts after normal cluster authentication rather than starting replacement experiments.

Frozen paper/evaluation copies retain their original bytes, hashes and source paths in the artifact store. They are provenance, not additional editable copies. PDF, bibliography output, logs, caches and preview images remain external; this directory contains only maintained authoring inputs and provenance/build receipts.
