# GeoAE continuation context

For a new Codex chat in this project, read [the project chat index](docs/CHAT_HISTORY.md) and the relevant main transcript before continuing research. Use [current setup/research status](docs/LAPTOP_SETUP_STATUS.md), [the current comparison CSV](runs/nonlinear_decoder_validation/deltaai/review/core_comparison.csv), and [project memory](docs/notes/MEMORY.md) for the active state. Re-read result files before quoting scientific numbers; historical chat and memory can be stale.

This Mac is the development home base. Keep large datasets, model loading, activation extraction, and GPU computation on Delta or Argo. Direct cluster access and GeoAE's supported CPU setup passed; see the setup status for evidence and the [handoff](docs/LAPTOP_HANDOFF.md) for paths. Use `bash scripts/remote/setup_laptop.sh --smoke-only` when CPU verification is relevant.

Opening the project or recovering context does not start experiments. Preserve existing remote source snapshots, manifests, hashes, and cluster data paths. The research root and older `../code/GeoAE` Git checkout differ; consult the setup status before reconciling them.

The two archived main chats and current Mac setup chat are assigned to the native GeoAE Codex project; helpers remain linked in the project index. The [import manifest](../work_context/codex_project_chat_import.json) records verification, and the [importer](scripts/remote/import_project_chats.py) defaults to a preview. The original frozen exports remain preserved in `../work_context`; the native migration chat additionally includes 16 later messages from the [visible-message refresh](../work_context/jetstream_visible_chat_refresh.json). See the project index for cutoff and import limitations.
