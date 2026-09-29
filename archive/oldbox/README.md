# Old-box archive

Outputs and tooling from the pre-Delta machine (and the first Delta attempt with the old
sheets), moved here on 2026-09-29 so the live tree only holds round-2 work. Nothing here is
read by the current pipeline.

- `eval_out/*.json`, `results/*.json|tsv` — old-box eval and intervention results
  (the numbers quoted in `docs/token_bypass_story.md` and `docs/notes/`).
- `eval_out/run_*.sh`, `eval_out/summarize_*.py` — the old per-arm command sheets and
  summarizers. They write fixed names and skip when an output exists, which is why round 2
  uses `eval_out/run_delta_evals.sh` + `eval_out/summarize_delta.py` instead.
- `PROBE_BIASBIOS_BALANCE_PHASED_COMMANDS.md`, `STEERING_DB14_COMMANDS.md` — old command notes.

Old-box numbers are a different replicate on a different corpus sample: compare them with
Delta runs only as margins (bypass minus base), never raw.
