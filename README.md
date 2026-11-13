# When Does a Small Model Pay?

A pre-registered study of escalation, harness routing and delegation for agentic small language models.
19 open models from 0.23B to 27B active parameters, four agentic tracks, 221,493 logged episodes, 23 pre-registered
hypotheses.

**Paper:** [PDF](docs/paper.pdf) · **Project page:** https://mawwlle.github.io/slm-agents/

## Findings

- A 4B model is a cheap substitute: Qwen3.5-4B solves 66% of the multi-turn tasks a 27B model solves, at a 7.4×
  lower cost per task. Below about 1.5B active parameters replaceability collapses.
- Harness fixes are model-specific. Of two findings from twelve models, the one about instructions replicated on seven
  new models and the one about worked examples did not. The tool-call parser alone moved Gemma-4-26B from 15% to 88%.
- A 27B planner levels small executors (a 350M model goes from 0% to 43% of multi-turn tasks), but it is 96% or more
  of the compute, and delegation does not pay.
- Prompted small planners do not hold the role: a 2.6B planner breaks the protocol in 91% of its turns. A router that
  plans only after a failure beats planning always by 17.8 pp at 59% of the cost; a gain over the best single model
  is not shown.

Of the 23 hypotheses, 6 hold, and all 6 survive a Holm correction. Every superiority rule was also run on 50
label-shuffled copies of the data, and a rule that passed more than 5% of them does not count.

## Layout

```
ear/         the harness: model client and tool-call parser, the agent loop, delegation, the four tracks
skills/      the per-track how-to files of the skill conditions
prereg/      the five pre-registrations, each with its dated log of changes
analysis/    prereg/ — the pre-registered tests as frozen; claims/ — one short script per claim in the paper
data/        tasks/, memory/ (worked examples), results/*.json, episodes.json (the manifest of the episode logs)
paper/       the LaTeX source; macros.py, tables.py and figures.py generate every number, table and figure
docs/        the project page (GitHub Pages); paper/site.py writes its data
tests/       the agent loop and delegation with scripted models
```

## Reproduce

Python 3.12; the harness uses only the standard library, the analysis needs `requirements.txt`.

```sh
pip install -r requirements.txt
python data/fetch.py                     # the episode logs, about 1 GB, checked against episodes.json
```

The episode logs are being uploaded to Hugging Face; `DATASET` in `data/fetch.py` gets its address then. Until
then, `data/results/` holds every result the paper uses, and everything from `paper.macros` on runs without them.

The analysis, from the episodes to every number in the paper:

```sh
python -m analysis.prereg.analyze        # stage 1        → data/results/prereg-stage1.json
python -m analysis.prereg.hr             # stage 2        → prereg-stage2.json
python -m analysis.prereg.stage3         # phase L, stage 3 → prereg-stage3.json
python -m analysis.prereg.cp             # stage 4        → prereg-stage4.json
python -m analysis.prereg.verdicts       # all 23 verdicts, Holm → verdicts.json
for c in c1_replace c2_parser c3_harness c4_delegate c5_cheap; do python -m analysis.claims.$c; done
python -m paper.macros && python -m paper.tables && python -m paper.figures
cd paper && tectonic -X compile main.tex --outdir build     # aee.tex is the anonymous submission
cd .. && python -m paper.site            # the page's data and PDF → docs/
```

New episodes: serve a model with an OpenAI-compatible server (we used vLLM 0.33) and run

```sh
python -m ear.run --model Qwen/Qwen3.5-4B --label qwen3.5-4b --base-url http://localhost:8000/v1 --active-b 4.2 \
    --tracks code,tools,shell,multiturn --limit 3 --out data/episodes/qwen3.5-4b/qwen3.5-4b.jsonl
```

`python -m pytest tests` checks the loop without a model.

## Credits

The multi-turn simulators in `ear/tracks/bfcl_api/` are copied unchanged from the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla) (Apache-2.0). Tasks come from MBPP,
BFCL v3 and BFCL v4.

## Citation

```bibtex
@misc{dunaev2026smallmodel,
  title  = {When Does a Small Model Pay? A Pre-Registered Study of Escalation,
            Harness Routing and Delegation for Agentic Small Language Models},
  author = {Dunaev, Maxim},
  year   = {2026},
  note   = {Preprint}
}
```
