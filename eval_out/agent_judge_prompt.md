# Agent judges for the head-to-head cluster comparison

Claude agents judge the blind pairs from `eval_out/run_interp.sh <arm> pair_export` (no API spend).
On 2026-10-01 Opus agents were a far better judge than the API models on the same items: real
clusters beat fake ones 95–100% of the time (Flash-Lite / GPT-4o-mini 75–88%), and the same cluster
won in both A/B orders 96% of the time (72% / 51%).

## Running it

1. `bash eval_out/run_interp.sh <arm> pair_export` (on the box) writes
   `results/delta/<arm>/interp/agent_judge/batches/batch_NN.md` (75 items each) and `key.json`.
2. Copy ONLY `batches/` to a shared path the agents can read — under `/projects/...`, not `/tmp`:
   `/tmp` is per login node, and the agents may run on a different node than your shell.
   Never copy `key.json` there: it names the spaces.
3. Spawn one agent per batch (model: opus), with the prompt below. Run at most 3–4 at once: ten
   parallel Opus agents hit the subscription session limit within ~5 minutes (730/750 items were
   still answered, because agents append after every chunk).
4. Copy `answers_*.jsonl` back to `results/delta/<arm>/interp/agent_judge/answers/` on the box and
   run `bash eval_out/run_interp.sh <arm> pair_import` -> `interp/pairwise_agents.json`.
   Missing items are reported and simply left out.

## Prompt (replace INPUT and OUTPUT)

You are a careful human-style annotator judging clusters of text snippets. This is a blind rating task.

Your input file: INPUT
Your output file: OUTPUT

The top of the input file explains the task and the scoring rubric; follow it exactly. The file holds
75 items (each starts with a line "=== ITEM qNNNN ===") and is about 1,600 lines long. Read it with
the Read tool in chunks of about 250 lines (offset/limit) so that you actually see every snippet.

Rules:
- Judge every item yourself by reading the marked «tokens» in their context. Do not write or run code
  to score items, and do not use shortcuts such as counting repeated words.
- Judge each item on its own merits. Items are unrelated and in random order.
- After each chunk, append the answers for the items you finished to the output file, one JSON object
  per line, exactly in the format the file header gives, including the "id". You may use Bash only to
  append lines (e.g. cat >> file <<'EOF') and to count lines. Every line must be valid JSON; "level"
  must be one of the listed levels; the comparisons must be "A", "B", "tie" or "neither".
- Do not open any file other than your input file and your output file.
- At the end the output file must contain exactly one line per item (75 lines) and nothing else.

When finished, reply with only: "done", the number of lines in the output file, and the ids of any
items you could not judge.
