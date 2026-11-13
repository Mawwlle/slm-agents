# Pre-registration — Stage 3: Plan & Delegate (frozen 2026-11-09 14:54 UTC, before any stage-3 episode)

## Claim
**H-PD.** A big model that only plans and checks, with small models executing each step under a simple harness,
reaches the quality of the big model working alone at a fraction of its cost — on tasks that really have steps.

Why it might hold: stage 2 showed small models use examples, not instructions. A plan turns a task into concrete,
almost template-like steps ("call `grep` on this file with this pattern"), which is close to an example; and the
planner can check each step and retry it cheaply, which matters because small-model failures are largely the
sampling lottery (stage 1, A2). Why it might not: such systems are reported to be planner-limited
(arXiv 2601.11327), and qwen3.8 is expensive mostly through its thinking tokens, so a verbose planner eats the saving.

## Roles
- **Planner**: qwen3.8 (gateway), thinking off, ≤ 1024 tokens per call. Variant: gemma-4-26B-A4B (gateway, 3.8B active).
- **Executor**: a small model under the *executor harness*: the system prompt holds only the current step from the
  planner, only the tools that step names, and up to 2 worked examples of a call to the same tool (from qwen3.8's
  successful train trajectories). At most 3 model turns per step (call → result → optional correction).
- **Closed loop** (main): the planner sees the task and tools and emits one step at a time as
  `{"step": "...", "tools": [...]}` or `{"finish": ...}`; after each step it sees the executor's calls and the tool
  results. Planner turns are capped at the track's step limit.
- **Open loop** (ablation): the planner emits the whole plan once; the executor runs it; no checking.

## Tracks
| track | tasks | why |
|---|---|---|
| BFCL v4 multi-turn base (new) | 200 | stateful API simulators, several calls per turn: steps are real |
| shell (stages 1–2) | 50 | multi-step by construction |
| tools: BFCL v3 single-turn (stages 1–2) | 800 | control: the planner does nearly all the work |
| code: MBPP (stages 1–2) | 497 | control: one function, little to delegate |
Splits as before (sha256 of the id). The decomposable tracks (multi-turn, shell) carry the hypotheses; the two
controls are reported to show where delegation does *not* help.

## Runs (all splits, temperature 0.6)
- Closed loop, planner qwen3.8, executors: LFM2.5-350M, LFM2.5-1.2B, LFM2.5-2.6B, LFM2.5-8B-A1B, gemma-4-E2B, gemma-4-E4B,
  Qwen3.5-2B, Qwen3.5-4B, FunctionGemma-270M.
- Closed loop, qwen3.8 as both planner and executor (P→B): separates "a good plan" from "delegation".
- Planner gemma-4-26B-A4B with executors LFM2.5-1.2B and Qwen3.5-4B.
- Ablations on LFM2.5-1.2B and Qwen3.5-4B: open loop; executor without step examples.
- Solo baselines on the multi-turn track for every model of stages 1–2 and phase L (`base`, `memory`).
About 32k episodes and ~120k short planner calls; roughly 1–1.5 days.

## Cost
FLOPs = 2·N_active·tokens, counted separately for the planner and the executor; the planner share is reported.

## Hypotheses (decomposable tracks = macro over multi-turn and shell; test split; 95% paired task bootstrap)
| ID | Claim | HOLDS if |
|---|---|---|
| PD1 · delegation keeps quality | the best executor ≤ 4B active is as good as qwen3.8 executing | Q(P→S) − Q(P→B) has CI lower bound > −5 pp |
| PD2 · delegation is cheaper | the P→S frontier beats the frontier of single models with their stage-2/L harnesses | at ≥ 1 budget, best P→S (chosen on fit) − best single pair ≥ 2 pp with CI > 0 |
| PD3 · how small an executor | — | smallest executor (by active parameters) with Q(P→S) − Q(P→B) CI lower bound > −5 pp; predicted ≤ 1.5B active |
| PD4 · planner-limited | swapping the planner moves Q more than swapping the executor | |Q(qwen3.8 plans) − Q(gemma plans)| − |Q(4B executes) − Q(1.2B executes)| > 0 with CI > 0 |
| PD5 · examples at step level | the step examples help the executor | Q(with examples) − Q(without) CI > 0 for both ablation executors |

Calibration: PD1–PD5 are paired differences of means with pairs chosen on fit (as HR2, whose rule passed 0% of 50
shuffled copies); the cascade machinery that failed HR3's calibration is not used.

## Log
- 2026-11-09 15:05 UTC (first written as 16:10 by mistake; commit b533316 is 15:06) — **implementation choices, before any stage-3 episode.**
  - *Multi-turn track* (`ear/tracks/multiturn.py`): the 8 API simulators are copied unchanged from bfcl-eval
    2026.3.23 (`ear/tracks/bfcl_api`); calls are executed and graded without BFCL's global instance cache, with its
    rules: after each request the public state must equal the ground truth's, the ground truth's results must be
    among the agent's results so far, a request with ground-truth calls needs a call and the 3 requests without any
    must get none. Score = share of requests passed before the first failure; pass = all. Checked before any model:
    replaying the ground truth through the environment scores 1.0 on all 200 tasks, doing nothing 0.0, and the same
    holds through the solo harness and through the delegation loop with scripted models. Splits by sha256 of the id:
    117 test, 52 train, 31 screen.
  - *Limits*: 10 steps and 24 000 completion tokens per user request on the multi-turn track (BFCL allows 20 steps);
    the other tracks keep their stage-1 limits. Under delegation the planner's turns count against the step limit
    and the token budget covers planner and executor together.
  - *Planner*: thinking off, 1024 tokens, the prompt in `ear/delegate.py`; a reply that is not one JSON object gets one
    correction message and counts as a planner turn. In the open loop a later request sees earlier requests and plans,
    never results.
  - *Executor*: each model with its stage-1/L serving and sampling settings (thinking as the model defaults); its
    `finish` ends the step, not the request. In P→B qwen3.8 executes with its stage-1 settings (thinking on).
  - *Step examples*: `data/memory/examples.json` from qwen3.8's passed base train episodes, calls that parse cleanly
    with an ok tool verdict, fixed order by sha256, a task never sees its own call. On the multi-turn track the bank
    is built after qwen3.8's solo multi-turn run, the first stage-3 run. On the tools track names rarely repeat, so
    most steps get no example there: that track is a control.
  - *Solo `memory` on multi-turn*: built as in stage 1 (3 nearest passed train episodes of qwen3.8 by bge-m3), after
    the same qwen3.8 run.
- 2026-11-10 15:20 UTC — **what happened while running, no rule changed.**
  - *Gateway access*: delegation runs start in the executor's sandbox, which had no gateway credentials until the
    operator's [credentials file] was uploaded at 2026-11-09 22:50 UTC; the first stage-3 delegation episode is after that.
  - *Gemma 26B as planner*: in about a quarter of its turns it calls a tool itself in its own syntax
    (`call:write_solution{...}`) instead of writing a step, and 837 more replies are JSON without `step` or `finish`.
    Both get the pre-registered correction message; nothing was changed. PD4 therefore measures, in part, whether
    Gemma can hold the planner role at all. An exploratory variant that turns such a call into a step would be
    reported apart from PD4, if run.
  - *Context*: a solo episode whose conversation outgrows the 32 768-token window ends as `agent_failed` (Qwen3.5-0.8B-wsp
    1, Granite 1, MNN-Pocket 7 of 400). The stage-3 analysis keeps these as failed: they are the model's own doing,
    and they are the only `agent_failed` episodes of stage 3. (The stage 1–2 loader drops every `agent_failed`
    episode; an earlier version of this entry wrongly said the two treatments were the same.)
  - *Pending*: gemma-4-E4B (moved to the qwen3.5-2b sandbox after its GPU stayed full for hours; same serving
    settings), Qwen3-VL-8B's solo multi-turn, and FunctionGemma-270M, which needs its own call format in the parser
    (a new parser version: every recorded step is re-parsed to check nothing else moves, as for v5): undecided, will
    be logged.
- 2026-11-10 16:50 UTC — **FunctionGemma-270M dropped, by the user's decision.** Running it would need a parser version
  for its own call format; with LFM2.5-350M at −37 pp from P→B, a 270M executor would not move PD1 or PD3. Stage 3
  therefore has 8 small executors, not 9. gemma-4-E4B ran in the qwen3.5-2b sandbox with its phase-L serving settings.
