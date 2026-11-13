# Pre-registration — Phase L: new small models as an out-of-sample check (frozen 2026-11-09 11:00 UTC, before any phase-L episode)

Stage 2 (PREREG-HR.md) left three findings that were read off the same 12 models they describe: examples (memory)
help every model, instructions (a skill, delivered either way) help only the big ones, and a skill can make a small
model *cheaper*. HR5 turned them into a router and showed that the frozen form had a hole. Phase L runs 7 models
that were not in stages 1–2, so the findings and the corrected router meet data they were not fitted on.

## Models (replace MNN-Pocket-0.3B, which solves ≤ 5% of tasks under any harness)
| model | total | active (FLOPs) | source of the active count |
|---|---|---|---|
| LiquidAI/LFM2.5-230M | 0.23B | 0.23 | dense |
| LiquidAI/LFM2.5-350M | 0.35B | 0.35 | dense |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | 1.2 | dense |
| LiquidAI/LFM2.5-2.6B | 2.69B | 2.69 | dense, model card |
| LiquidAI/LFM2.5-8B-A1B | 8.3B | 1.5 | MoE, model card |
| google/gemma-4-E2B-it | 5.1B | 2.3 ("effective"; per-layer embeddings are lookups) | model card |
| google/gemma-4-E4B-it | 8B | 4.5 ("effective") | model card |
FunctionGemma-270M is not run alone: in a 9-task smoke it declined every code and shell task and wrote its own call
format. It is kept as an executor candidate for stage 3.

## Runs
1347 tasks, all splits, temperature 0.6, stage-1 limits; conditions `base`, `memory`, `skill`, `skill_inline`,
`skill_inline+memory`. Parser v4 unchanged (smoke: LFM2.5 writes the harness's `<tool_call>` JSON). If a model needs
a parser change, it is logged here before its full run, and stage 1–2 data are re-parsed to show they do not move.

## Hypotheses (lifts are paired with `base`, macro over tracks, 95% task bootstrap)
| ID | Claim | HOLDS if |
|---|---|---|
| L1 · examples help small models | replicates stage 2 | `memory` lift CI > 0 for ≥ 5 of the 7 models |
| L2 · instructions do not | replicates stage 2 | `skill_inline` lift CI > 0 for ≤ 2 of the 7 models, and the mean over the 7 is below the mean `skill_inline` lift of the 3 gateway models (CI of the difference < 0) |
| L3 · corrected router | HR5's router with a fallback ladder and the skill as a cost reducer, on the 7 new models + the 3 gateway models | see below |
| L4 · skill as a cost reducer (descriptive) | | per model: FLOPs/task of `skill` and `skill_inline` against `base`, with CIs |

**L3 rule** (written before the data): fit-split mean FLOPs/task C. For each model a ladder, best first:
big (≥ 25B): `skill_inline+memory` → `skill` → `base`; small: `memory` → `base`, where `base` is replaced by `skill` for
any model whose C(skill) < C(base) on fit. At budget B, h(m, B) = the first ladder rung with C ≤ B; the router takes the
best fit Q among (m, h(m, B)). The search router takes the best fit Q among all pairs with C ≤ B. 16 log-spaced budgets,
4e12–1.6e14. On test: d(B) = Q(rule) − Q(search), paired task bootstrap. **HOLDS** if the CI lower bound of d(B) > −5 pp
at every budget (the 2 pp margin of HR5 is out of reach at this test size: CIs are about ±4.5 pp) and the mean of d
over budgets ≥ −2 pp.

Calibration: L1/L2 are per-model paired means (the bootstrap is calibrated for these, cf. HR2 0% false PASS); L3 is a
non-inferiority bound and is reported together with the share of budgets where both routers pick the same pair.

## Log
- 2026-11-09 14:40 UTC — **parser v5, before any phase-L analysis.** The LFM2.5-2.6B smoke scored 0/9 because every
  step was "broken": the model ignores the harness's `<tool_call>{json}` protocol and writes its native
  `<|tool_call_start|>[name(key=value, …)]<|tool_call_end|>`, with sensible calls inside. Parser v5 reads that shape
  (a Python call list, arguments as literals) and nothing else changes. Re-parsing every step of stages 1–2
  (443 772 steps, 59 files) with v4 and v5 gives identical calls and broken flags on all of them: the marker never
  occurs there. It does occur in phase L: LFM2.5-230M wrote it in 124 steps, 80 of which now parse differently
  (43 of 6735 episodes). Keeping the unaffected episodes and rerunning only the affected ones would mix a conditional
  and an unconditional sample, so, as with v4 in phase M, **every LFM2.5 phase-L run is redone from scratch under
  v5** (230M, 350M, 1.2B, and the 64 episodes of 2.6B). The v4 outputs are kept as `*-L-v4.jsonl` for a
  descriptive parser-sensitivity note. The 1.2B smoke failure before that was not the model: vLLM found 3.8 of
  the 7.1 GiB it asked for free on the shared GPU; the driver now retries the start.
- 2026-11-09 15:45 UTC — **non-JSON argument values.** Three LFM and Gemma episodes crashed with "Object of type
  ellipsis / complex is not JSON serializable": the Python-literal readers (LFM's native calls, and Gemma's since
  v4) passed `...` or `1j` through, and `harness.signature` serialises every step's calls, so such an episode never
  reached the output, in any stage. The readers now turn such values into strings (`'...'`, `'1j'`). Re-parsing all
  461 158 recorded steps with the old and new code gives identical results, so no recorded episode changes; the
  crashed episodes are simply rerun. Before this fix a crash dropped an episode and the resumed run redrew it, as
  in stages 1–2 (≤ 3 per 6735 here).
