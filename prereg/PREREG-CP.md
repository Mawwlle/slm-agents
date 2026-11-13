# Pre-registration — Stage 4: Cheap Planner with a Router vs a Tiny Stack (frozen 2026-11-11 13:55 UTC, before any stage-4 episode)

## Claim
Stage 3 found that a plan levels executors (with qwen3.8 planning, every executor from 1.5B active solves 55–59% of
multi-turn tasks; LFM2.5-350M goes from 0 to 43%), that a strong executor gains almost nothing from a plan
(Qwen3.5-4B 53.8 → 55.6%), and that the scheme does not pay: 95% of the FLOPs are the 27B planner's, and at the same
cost Qwen3.5-9B or 4B alone do as well or better. Two ways to make delegation cheap are compared here:

- **A — cheap planner, used selectively.** A model M first solves the task alone; only when a router flags a likely
  failure is the task solved again as Qwen3.5-9B (or 4B) planning and M executing.
- **B — tiny stack, always on.** LFM2.5-2.6B plans and LFM2.5-350M executes, on every task.

What the literature predicts. Planner/executor splits of small models work when the roles are trained: α-UMi
(arXiv 2401.07324) fine-tunes planner, caller and summarizer; Plan-and-Act (2503.09572) and TinyAgent (2409.00608)
fine-tune the planner; ToolOrchestra (2511.21689) trains an 8B orchestrator with RL. Prompted, untrained small planners
are not shown to hold the role, and in stage 3 Gemma-4-26B-A4B did not (it called tools itself in about a quarter of
its turns). Cascades that answer with the cheap model and escalate only when in doubt keep quality at a fraction of
the cost (FrugalGPT 2305.05176, AutoMix 2310.12963, RouteLLM 2406.18665; the "SLMs by default, LLMs on demand"
position of 2506.02153), and in stage 1 the first steps of a trajectory predicted failure with AUC 0.86. So A is
predicted to beat B.

## Systems
- D(P→E): the closed loop of `ear/delegate.py` (PREREG-PD): planner P, thinking off, ≤ 1024 tokens per call; executor
  E with step examples (`data/memory/examples.json`), ≤ 3 turns per step. Planners and executors are local vLLM servers
  in the same sandbox (P on :8001, E on :8000), with their stage-1/L serving and sampling settings.
- Solo M: the stage-1 harness, condition `base`.
- R(M; P): M runs solo; when the router fires, the task is solved again from scratch as D(P→M) (the simulators are
  reset, so this is a fresh episode). Evaluated offline from the solo and D episodes of the same tasks:
  - **R-obs** (no fitting): fires when the solo episode ends in an observable failure — `timeout`, `loop`,
    `broken_tool_call`, `model_stopped` or `agent_failed`. Cost = the whole solo episode + the D episode.
  - **R-det**: a logistic detector on the features of the first k = 3 solo steps (`analysis.load.step_features`, the
    stage-1 A1 model: median imputation with indicators, standard scaling, logistic regression C = 0.5), cross-fitted
    over 5 task folds; it fires when p(fail) ≥ τ. Cost = the k-step prefix + the D episode if it fires, else the whole
    solo episode. τ is chosen per budget on the fitting folds.

## Tasks
| track | tasks | note |
|---|---|---|
| BFCL v4 multi-turn base | 200 | as stage 3 |
| BFCL v4 multi-turn miss_param (track `multiturn_miss_param`) | 200 | new; a request that lacks a parameter must get no call (the stage-3 grader rule) |
| BFCL v4 multi-turn long_context (track `multiturn_long_context`) | 200 | new; scenarios loaded with `long_context=True` |
| shell (stages 1–2) | 50 | as stage 3 |
`multi_turn_miss_func` is left out (its tool list changes within an episode). The two new categories come from the same
bfcl-eval 2026.3.23 package and simulators, built by `tools/build_multiturn.py` (which rebuilds
`data/tasks/multiturn.jsonl` byte for byte). Checked before any model: replaying the ground truth scores 1.0 on all
600 multi-turn tasks, and doing nothing scores exactly the share of leading requests whose ground truth is empty —
0 on base and long_context; 81 miss_param tasks open with such a request, so doing nothing gets partial credit there
and never a pass. The new tracks use the multi-turn track's step examples (same tools). All 650 tasks are used; fitting (detector, τ, choice among routers) is cross-fitted
over 5 task folds stratified by track, so every task is scored by a rule fitted without it.

## Runs (temperature 0.6, one repeat)
- Solo `base` on the 400 new tasks: Qwen3.5-4B, Qwen3.5-9B, LFM2.5-2.6B, LFM2.5-350M (the multi-turn base and shell
  solo runs exist).
- D on all 650 tasks: D(9B→350M), D(9B→4B), D(9B→2.6B), D(4B→350M), D(4B→2.6B), **B = D(2.6B→350M)**, and D(2.6B→2.6B).
About 1.6k solo and 4.5k delegated episodes.

## Measures
Q = macro pass rate over the four track rows above. Cost = FLOPs per episode, 2·N_active·(prompt + completion tokens),
planner and executor counted separately, as in stages 1–3. CIs: 95% paired task bootstrap within track
(10 000 draws). Protocol violation: a planner turn whose reply is not one JSON object with `step` or `finish`.

## Hypotheses
| ID | Claim | HOLDS if |
|---|---|---|
| CP1 · a small planner holds the role | LFM2.5-2.6B can plan for a smaller executor | planner protocol-violation rate in B ≤ 10% **and** Q(B) − Q(LFM2.5-2.6B solo) > 0 with CI > 0. Predicted: does not hold |
| CP2 · a cheap planner lifts a tiny executor | | Q(D(9B→350M)) − Q(350M solo) > 0 with CI > 0. Predicted: holds. Descriptive: D(9B→350M) against D(qwen3.8→350M) of stage 3 on multi-turn base |
| CP3 · selective planning is as good as always planning | | for the better of R-obs / R-det (chosen on the fitting folds): Q(R(4B; 9B)) − Q(D(9B→4B)) has CI lower bound > −3 pp at a mean cost ≤ 60% of D(9B→4B)'s. Predicted: holds |
| CP4 · selective planning beats a single model | | at ≥ 1 of 16 log-spaced budgets spanning the fitting-fold costs of all systems, the best R (over M ∈ {Qwen3.5-4B, LFM2.5-2.6B}, P ∈ {9B, 4B}, signal and τ, chosen on the fitting folds) − the best solo pair (Qwen3.5-4B, Qwen3.5-9B, LFM2.5-2.6B, LFM2.5-350M `base`) ≥ 2 pp with CI > 0. Predicted: uncertain |
| **CP5 · A against B** | the user's question | at the budget equal to B's mean cost, the best R chosen on the fitting folds under that budget − Q(B) > 0 with CI > 0. Predicted: holds |

Calibration: the CP3–CP5 selection (router, signal, τ) is rerun on 50 copies in which, per task, the labels of the
compared systems are exchanged with p = 1/2; at most 5% false PASS is required for a verdict to stand. The stage-2
cascade machinery is not reused.

Caveat stated in advance: solving a task "again from scratch" is only possible because the simulators reset; a real
task with side effects would need the plan to start from the solo run's state.

## Log
- 2026-11-11 13:55 UTC — **frozen; implementation choices, before any stage-4 episode.** Queues in `tools/queue3.py` (cp-9b,
  cp-lfm); planner on :8001 beside the executor (`tools/serve.sh` PORT, `tools/probe.py --port`). Beside the 9B
  planner, Qwen3.5-4B is served with GPU util 0.30 and a 3 GiB KV cache (stage 1: 0.36, 4 GiB) so both fit one 48 GB
  card; this changes concurrency, not sampling. Solo runs on the new tracks use stage-1 serving. Run files:
  `<executor>-cpsolo` (solo, new tracks), `<executor>-cp9` / `-cp4` / `-cpl` (planner Qwen3.5-9B / Qwen3.5-4B /
  LFM2.5-2.6B).
- 2026-11-12 08:36 UTC — **R(Qwen3.5-4B; Qwen3.5-4B) dropped from CP4, before any CP3–CP5 result was seen.** The CP4 router set
  (M ∈ {Qwen3.5-4B, LFM2.5-2.6B} × P ∈ {9B, 4B}) includes R(4B; 4B), which needs D(4B→4B); the run list of this
  pre-registration does not include that run, so the router has no data (the analysis stopped on it). CP4 and CP5
  use the three routers with data: R(4B; 9B), R(2.6B; 9B), R(2.6B; 4B). No run was added.
- 2026-11-12 08:37 UTC — **two problems with the rules as written, found when the first analysis ran (CP3–CP5 point results had
  been seen; the 50-copy calibration had not).** (1) *CP3's calibration cannot be met by any non-inferiority rule.*
  CP3 claims R is at most 3 pp worse than D; on label-shuffled copies the two are equal, so the claim is true there
  and a pass is not a false pass. The shuffle checks superiority claims (CP4, CP5); for CP3 its pass rate is reported
  but does not void the verdict. (2) *CP5 cannot be evaluated at B's cost.* B costs 2.25e14 FLOPs per episode because
  its planner breaks the protocol and episodes end early; every router costs at least its solo model (≈ 2.4e14), so
  no router fits that budget and the rule returns NOT SHOWN by construction. This is reported as such; the
  comparison at B's cost is made descriptively (LFM2.5-2.6B alone costs 2.44e14, 8% more than B), with the best
  router at the nearest budgets beside it. Also fixed: an intersection call that crashed on three routers.
- 2026-11-12 08:38 UTC — **results (data/results_cp.json, 50 calibration copies).** CP1 NOT SHOWN: B's planner breaks the protocol
  in 92% of turns, and B − LFM2.5-2.6B alone = −37.9 pp [−42.1, −33.6]. CP2 HOLDS: D(9B→350M) − 350M alone =
  +20.4 pp [+16.8, +24.1]. CP3 HOLDS: R(4B; 9B) − D(9B→4B) = +17.8 pp [+13.2, +22.4] at 59% of D's cost (the plan hurts
  this strong executor and the router keeps it mostly alone; the shuffle passes 12%, moot for non-inferiority, see
  above). **CP4 NOT SHOWN, voided by calibration:** the best router beats the best single model by +3.1 pp
  [+1.1, +5.4] at 3.35e14 FLOPs and by +1.9 pp at two more budgets, but the rule passes 14% of shuffled copies (> 5%),
  so the gain is not shown; at the highest budget Qwen3.5-9B alone is ahead by −3.9 pp. CP5 NOT SHOWN, not evaluable
  (no router fits B's cost; see above).
