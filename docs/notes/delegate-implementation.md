---
name: delegate-implementation
description: "Main session = planning/analysis partner; hand code implementation to cheaper subagents (sonnet/haiku) with a precise spec, then review their work"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b1c5c1c3-1bd1-42d3-9c11-49132df71932
  modified: 2026-09-26T05:52:49.167Z
---

From 2026-09-26 the user wants the main session to be "the main mind that I chat and plan with", and implementation work done by cheaper agents wherever possible.

**Why:** cost — the main model is expensive; routine coding/smoke-testing doesn't need it.

**How to apply:** design the experiment yourself (targets, arms, metrics, confounds, scoring), then spawn an Agent with `model: "sonnet"` (haiku for trivial edits/searches) and a self-contained spec: files to create/modify, exact definitions, guards, tests, smoke-run limits, and "don't modify existing behaviour / no commits / never print .env". Review the diff and smoke output yourself before handing the user the command sheet ([[report-base-vs-bypass-only]] still applies to how results are reported). Long GPU runs are still handed to the user as copy-paste sheets.
