# Pre-registration — Harness routing (frozen 2026-11-08 22:30 UTC, before any phase-H episode)

Stage 2 of the project. Stage 1 (PREREG.md) is closed except for the memory wave on the small models.
Written after stage 1's results were read; what was already seen is marked below.

## Claim
**H-HR.** The best harness depends on the model. A router that picks a (model, harness) pair reaches higher
quality at the same cost than a router that picks only the model and gives every model one harness.

## Harness components (conditions in `ear/harness.py`, combined with `+`)
| component | what it does |
|---|---|
| `skill` | SKILL.md listed by name; the model loads it with `read_skill` (progressive disclosure) — stage 1 |
| `skill_inline` | the SKILL.md text in the system prompt, no `read_skill` |
| `memory` | 3 worked qwen3.8 trajectories on the nearest train tasks (stage 1's `data/memory/memory.json`) |
| `strict` | vLLM structural tag: after `<tool_call>` the decoder is held to the tools' JSON (types only); free text and thinking stay free. Local vLLM models only: the gateway does not guarantee it |
| `watchdog` | stage 1 |

## Phase H runs (all splits, 1347 tasks, temperature 0.6, same step limits and budgets as stage 1)
- 9 local models: `skill_inline`, `strict`, `skill_inline+memory`, `strict+skill_inline+memory`
- 3 gateway models: `skill_inline`, `memory`, `skill_inline+memory`
- Already available from stage 1: `base` (×2), `skill`, `watchdog` for all; `memory` for the 9 local models.

## Hypotheses and decision rules (test split; everything that is chosen is chosen on screen+train)
| ID | Claim | HOLDS if |
|---|---|---|
| HR1 · cross | Some component helps small models and not big ones, and some the other way | For a component c: d(c) = mean lift over small models (≤ 10B, local, without the WSP copy) − mean lift over the 3 gateway models, task-bootstrap 95% CI. HOLDS if at least one component has CI of d > 0 **and** at least one has CI of d < 0. `strict` has no big side and is reported only as a small-model lift. *Seen already: `skill` lift is ≈ 0 on small and +6.4 pp on big (stage 1).* |
| HR2 · per-model harness | A harness chosen per model beats the best single harness for all | On screen+train: H\* = the single harness with the best mean macro Q over models; h\*(m) = the best harness for model m among those whose mean FLOPs/task ≤ that of H\* for m. On test: ΔQ(m) = Q(h\*(m)) − Q(H\*) with a 95% task-bootstrap CI. HOLDS if the CI is > 0 for ≥ 2/3 of the models where h\*(m) ≠ H\*, and the mean ΔQ over all 12 models is > 0 (CI). |
| HR3 · retry with another harness | After the detector flags a run, a retry with the model's own harness beats a plain retry | Stage 1's offline cascades (`cascades` in `analysis/analyze.py`, k = 2, both big targets): policy `switch` = retry with h\*(m) instead of base. HOLDS if Q(switch) − Q(retry) > 0 with a 99% CI (Bonferroni over models) at ≥ 2 budgets of one family for ≥ 1 model, and REVERSED if < 0 the same way. |
| HR4 · pair router | The cost–quality frontier over (model, harness) pairs dominates the frontier over models with `base` | Budgets 1e13, 2e13, 5e13 FLOPs/task: best test macro Q among pairs with mean cost ≤ budget, minus the best among (model, base). The pair is chosen on screen+train, scored on test. HOLDS if ≥ 2 pp with a 95% task-bootstrap CI > 0 at ≥ 1 budget. |

Noise floor: the `base` repeat next to every effect, as in stage 1. Calibration: HR2 and HR3 rules applied to
50 label-shuffled copies must PASS on ≤ 5% of them.

## Changes after freezing
- 2026-11-08 22:50 UTC — **`strict` grammar, before any analysed phase-H episode.** The smoke runs showed that for
  models whose `<tool_call>` is a special token cut from the text (Granite 4, MNN) the trigger never fired, so
  `strict` did nothing for them; and under the grammar Granite spelled its end-of-sequence as text (`</s>`) and
  ran on into a fenced fake answer. The grammar now also opens on the bare `{"name": "` these models write, and
  `strict` requests carry stop strings for the textual EOS forms (`</s>`, `<|end_of_text|>`, `<|endoftext|>`,
  `<|im_end|>`). Smoke after the fix: Granite `strict` 0 broken steps of 18 (10 of 21 before). The phase-H runs
  already started on granite, MNN, qwen3.5-2b and qwen3.5-9b were set aside and restarted from scratch; the
  gateway runs have no `strict` and were not touched.
- 2026-11-09 09:30 UTC — **results** (`data/results_hr.json`, 137 393 episodes). HR1 NOT SHOWN (skill and skill_inline
  d < 0; no component with d > 0: memory helps both sizes alike). HR2 NOT SHOWN (H* = memory; 2 of 3 moved models
  win; mean +0.7 [−0.1, +1.5]); calibration 0% false PASS. HR3 HOLDS by the rule but **the rule failed calibration**:
  48% false PASS on 50 swapped copies (≤ 5% required), so HR3 is not claimed. The diagnostic shows single
  model × budget cells exceeding 99% CIs far more often than 1% under the null: the task bootstrap ignores that the
  detector threshold and the matched-cost point are picked on the same data. The same machinery underlies stage 1's
  A3, whose HOLDS is now marked provisional. HR4 HOLDS (2e13: +8.6 [+4.0, +13.8]; 5e13: +8.0 [+3.3, +12.2]).
  Post hoc, not pre-registered: no single harness for all models reproduces the pair frontier.

## HR5 — the three-rule router (frozen 2026-11-09 10:05 UTC, before it was computed)
Proposed after HR1–HR4 were read, so the rules come from the same data; the pair is chosen on screen+train and
scored on test, which limits but does not remove the circularity. The real out-of-sample check is phase L.
- **Rule harness** h_rule(m, B) for model m at budget B (fit-split mean FLOPs/task C):
  big model (≥ 25B, the 3 gateway models): `skill_inline+memory` if C ≤ B, else `skill`;
  small model: `memory` if C ≤ B, else `base`.
- **Rule router** at B: among the pairs (m, h_rule(m, B)) with C ≤ B, the one with the best fit macro Q.
- **Search router** at B: the best fit macro Q among all (model, harness) pairs with C ≤ B (as in HR4).
- Budgets: 16 log-spaced points from 4e12 to 1.6e14 FLOPs/task. On test: d(B) = Q(rule) − Q(search), paired task
  bootstrap, 95% CI.
- **HOLDS** if the CI lower bound of d(B) > −2 pp at every budget where both routers have a pair; the mean of d over
  budgets is reported next to it.
- 2026-11-09 10:20 UTC — **HR5 result: NOT SHOWN**, mean d = −3.8 pp over 16 budgets. Three causes, in order of size:
  (1) a hole in the frozen rules: a big model whose rule harness does not fit drops out instead of falling back
  to `base` (−39 pp at 1.75e13); (2) the cheap end of the frontier: a skill makes some small models *cheaper*
  (Granite with `skill` 6.8e12 FLOPs/task against 9.2e12 in `base`) and `skill_inline` lifts Qwen3-0.6B by
  +10 pp, which "skill only for big models" forbids (−10 and −12 pp at 5e12 and 8e12); (3) power: a paired
  difference between two different pairs has a 95% CI of about ±4.5 pp on this test split, so the −2 pp
  non-inferiority margin cannot be met even where the point difference is 0–2 pp. At budgets ≥ 2.2e13 the rule
  router equals the search or differs by −0.9…+2.0 pp, all CIs containing 0. Post hoc (not a test): with a
  fallback ladder (big: `skill_inline+memory` → `skill` → `base`; small: `memory` → `base`) the mean d is −1.5 pp.
- 2026-11-11 15:31 UTC — **HR3 statistics corrected after its calibration failed (commit 65febd7).** HR3 passed 48% of the 50
  label-shuffled copies, so its "HOLDS" could not stand. Two causes in the cascade code, neither in the rule as
  written: (1) each policy's detector threshold was the best one on test and then scored on the same tasks; (2) the
  switch and plain-retry policies were compared at their own best thresholds, so they sent different shares of tasks
  to the big model, and that, not what the second run was, separated them. Now thresholds are chosen on the fit
  split (screen + train) and scored on test, and switch is compared with retry at retry's threshold, which is what
  HR3 asks (what to retry with). A paired permutation p is reported beside each 99% Bonferroni CI; the decision rule
  is unchanged. Rerun on the stage 1–2 episodes only (the loader had begun to read later stages' files; fixed in the
  same commit): calibration false PASS 1 of 50 (2%), HR2 0%. **HR3 HOLDS**: for Qwen3.5-4B, switching to its own
  harness beats a plain retry at all six budgets, +1.4…+2.1 pp, every CI above 0, permutation p ≤ 0.001; Granite at
  one or two budgets. HR1, HR2, HR4, HR5 verdicts unchanged.
