# Pre-registration — Escalate, Augment or Repair? (frozen 2026-11-08, before any model run)

Written before the first episode. Changes after this point are listed at the end with a date and a reason.

## Hypotheses and decision rules
| ID | Claim | HOLDS if | Test |
|---|---|---|---|
| A1 | Label-free harness signals at step k≤3 predict the final failure of a small model | test-split AUC ≥ 0.75 **and** ≥ prompt-embedding AUC + 0.10 (paired bootstrap 95% CI of the AUC difference excludes 0.10) | logistic regression on step features (train split) vs logistic regression on bge-m3 prompt embeddings (same tasks, same split); pooled over small models, per track reported |
| A2 | ≥ 40% of small-model failures are obstacles, not missing capability | recoverable share ≥ 0.40, 95% CI lower bound > 0.30 | among (task, model) pairs failed in `base`, the share solved by `skill`, `memory` or `watchdog` for the same model (or by the WSP variant for Qwen3.5-0.8B), **net of the noise floor** (share solved by `base` repeat 2) |
| A3 | Lever-matched policy beats escalation-only at equal cost | ΔQ > 0 at matched cost with 99% Bonferroni CI at ≥ 2 of 3 budgets | phase P, test split |
| A4 | Skill lift vs size is ∩-shaped | lift at the peak size bin > lift at both the smallest bin (≤ 0.6B) and the largest (≥ 26B), each with 95% CI | `skill` − `base`, paired by task, per model |
| A5 | The WSP edit of Qwen3.5-0.8B lowers loops in agentic tasks and improves Q | loop share lower and Q higher than the base model, both 95% CI excluding 0 | same tasks, same conditions |
| A6 | An agent can find and remove a loop direction in another SLM (≤ 50 forward passes) | loop share −30% with Q not lower on held-out | descriptive, n small |

Calibration: the A1 rule is applied to 50 label-shuffled copies of the data; it must PASS on ≤ 5% of them, otherwise the rule is tightened before reading the real result.

## Data
- code: MBPP full test, 497 tasks (3 dropped: the reference solution fails its own tests: mbpp-180, 367, 493). Visible test = the first; graded on all three.
- tools: BFCL v3 — simple (every 2nd, 200), multiple 200, parallel 200, parallel_multiple 200. AST-style match against possible answers.
- shell: 50 generated tasks, 10 families × 5 seeds.
- Splits by sha256(task id) mod 100: screen < 15 ≤ train < 40 ≤ test.
- Checks validated: reference = 1, empty = 0 on every kept task.

## Models
Small (vLLM 0.33, bf16, in agentic sandboxes): MNN-Pocket-0.3B, Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-0.8B-WSP (RikiyaT), Qwen3.5-2B, granite-4.0-micro, Qwen3.5-4B, Qwen3-VL-8B-Instruct, Qwen3.5-9B.
Gateway: gemma-4-26B-A4B-it (3.8B active), qwen3-vl-30b (3B active), qwen3.8 (27B dense).
Sampling: temperature 0.6, top_p 0.95, the model's default thinking mode, 3072 tokens per step, 24k completion tokens per episode. Steps: code 8, tools 4, shell 10.

## Conditions
base · skill (listed, loaded with read_skill) · memory (top-3 qwen3.8 successful trajectories on *train* tasks of the same track, by bge-m3 similarity; for train tasks themselves the task's own trajectory is excluded) · watchdog (nudge on 2 repeats in a row, 2 broken calls in a row, or a cut-off step). base is run twice (repeat 0 and 1) as the noise floor.

## Failure kinds
'' (passed) · loop (≥ 2 consecutive repeated calls, or a cut-off step with > 30% repeated 8-word shingles) · model_stopped (cut off, not repetitive) · broken_tool_call · timeout (step limit) · budget · wrong_answer · agent_failed (API error: rerun, excluded).

## Cost
FLOPs = 2 × active params × (prompt + completion tokens), summed over steps (prefix caching ignored, as before); wall seconds reported too.

## Changes after freezing
- 2026-11-08 15:05 — **tool-call parser**, before any analysed episode: the smoke runs showed that Qwen3.5 writes
  `<tool_call><function=name>{json}` (and Qwen3-Coder-style `<parameter=…>`), and Gemma 4 writes
  `call:name{key: value}` with `<|"|>` string delimiters. The first parser took these for broken calls or for a
  final answer, which measures the harness, not the model. The parser now reads all three shapes and JSON with
  raw newlines inside strings. Runs started with the old parser (qwen3.5-0.8b, qwen3.8, gemma-4-26b-a4b) were
  stopped and their output set aside; they rerun from scratch.
- 2026-11-08 16:30 — **Gemma parser, second pass**: served through the gateway, Gemma 4 loses its `<|"|>` string
  tokens, so values arrive unquoted (`call:bash{command:ls -la\n}`, `key=value`). Such calls now parse (a value runs
  to the next `, key:` or the last `}`). Only the `call:` path changed; the gemma run was restarted from scratch.
  Other models' runs continue (their episodes never take that path; checked offline with `broken_shapes`).
- 2026-11-08 (third pass) — **bare JSON calls**: Granite 4 (and MNN-Pocket) lose their `<tool_call>` special token in
  serving and write bare `{"name": …, "arguments": …}`. Those now parse; the granite and MNN runs restarted from
  scratch. For models whose runs started earlier, episodes with a broken step of that shape are counted
  (`broken_shapes`) and rerun with the final parser if they exceed 2% of steps.
- 2026-11-08 19:00 — **parser v4, full rerun of phase M.** An audit of every broken step collected so far
  (22 904 episodes) showed that most "broken" calls were unambiguous calls in shapes the parser did not read:
  `{"function": name, …}` (qwen3.8: 19% of its steps), Python literals `True`/`None` inside JSON (Qwen3.5-0.8B),
  `{"tool_name": {args}}`, `{"name": "finish"}` without arguments, and typos such as `{"name="x"`. Parser v4 reads
  every shape where the tool is unambiguous: a malformed call is rebuilt only when exactly one of the task's tool
  names appears in it; such steps carry `recovered: true`, and every episode carries `parser: 4`. **All phase-M
  data so far is set aside** (`data/discarded-v3/`) and every model reruns from scratch on v4, so all models share
  one harness. The set-aside data is kept for a descriptive parser-sensitivity analysis (same tasks, v3 vs v4).
  Small models (≤ 2.3B) get 96 client workers and `--max-num-seqs 128` (throughput only).
- 2026-11-08 — gateway credentials: workers on the gateway read [credentials file], written by the operator
  (the agent's terminal does not get the gateway variables).
- 2026-11-08 21:30 — **phase M complete on v4**: 12 models × 5388 episodes, 0 `agent_failed`. One episode is missing
  (qwen3.5-2b, `tools/parallel_178`, skill): the decoder's Python-literal path returned a set, which crashed the
  JSON log (`TypeError: Object of type set is not JSON serializable`). The decoder now turns sets/tuples into
  lists (`_plain`); no parse result changes otherwise, so `PARSER_VERSION` stays 4. The episode is left out.
- 2026-11-08 21:30 — **memory wave**: `data/memory/memory.json` built from qwen3.8's v4 phase-M data
  (base r0, passed, train split: 182 worked examples; top-3 same-track by bge-m3, never the task itself).
  The 9 small models run `--conditions memory` in their own sandboxes after phase M. The agentic server was
  unreachable 19:40–21:25 UTC (network); runs inside the sandboxes were not affected.
- 2026-11-08 22:00 — **analysis changes, written down before the memory data came in** (A1–A5 had been read on phase M):
  - A1, prompt baseline: the embedding and the context columns (track, model, log size) went through one PCA, which
    can wash the context out and so weakens the baseline. Now PCA(64) is fit on the embedding alone (train split),
    and the context is appended unchanged; a context-only baseline is reported too. This makes the baseline
    *stronger*. The pre-registered pooled test is kept as the verdict. **Post hoc**, labelled as such: the same
    thresholds on the mean of within-model AUCs (`verdict_within`), since a router decides for one model at a time
    and the pooled AUC also rewards ranking models against each other.
  - A3: phase P (online) is replaced by **offline restart cascades** built from phase-M episodes. A restarted
    attempt is an independent episode, so its outcome and cost are exactly those of the phase-M episode of that
    condition (the first k = 2 steps of a stopped attempt are charged pro rata). The detector and the lever map are fit on
    screen+train, policies are scored on test. Escalation targets: gemma-4-26B-A4B and qwen3.8 (both named in the
    plan). Added: budgets relative to the small model's own cost (×1.25/1.5/2), because gemma-4 (MoE, 3.8B active)
    costs fewer FLOPs per task than the dense Qwen3.5 2–9B, so most "share of big" budgets are empty; and a `retry`
    control (second plain base sample instead of a lever) to separate "the lever" from "a second try".
    CIs: task bootstrap, 99%, Bonferroni over the 5 small models.
  - A6 was conditional on A5 (Q higher with the WSP edit); A5's Q part is not shown, so A6 is not run.
- 2026-11-08 23:55 UTC — **memory wave complete**: 9 local models × 1347 tasks, all keys unique, parser v4. Stage 1 is
  closed on 76 778 episodes; the analysis rules above did not change after the memory data came in.
- 2026-11-11 15:34 UTC — **A3 rerun with the corrected cascade statistics (commit 65febd7; see PREREG-HR's log for the cause).**
  The A3 cascades shared the code that failed HR3's calibration: each policy's threshold was the best one on test.
  Thresholds are now chosen on the fit split (screen + train) and scored on test; the rule (99% Bonferroni CI at
  ≥ 2 of 3 budgets of one family) is unchanged, and A3 compares cost-matched operating points by design, so levers
  and escalation keep their own fit-chosen thresholds. **A3 HOLDS** (unchanged): with qwen3.8 as the big model,
  levers beat escalation for Qwen3.5-4B at all six budgets (+3.2…+6.3 pp), Qwen3.5-9B at five (+2.6 pp), Granite at
  four; two cells go the other way (Granite with Gemma-26B at 100% of its cost, −15.6 pp; Qwen3-VL-8B at 50%, −7.6 pp).
  A1, A2, A4, A5 verdicts unchanged.
