# Experiments

Self-contained experiment logs for edge-VLM procedure grounding. One folder per experiment, each with a re-runnable `run.py`, a captured `config.json`, raw `results/`, and a `REPORT.md` card. Inspired by [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) — small hackable scripts, captured config, a report card, an accumulating log — minus (for now) the autonomous proposer loop and automatic scoring. We're in interactive-exploration mode; this is the foundation an optimization loop would slot into later.

## Leaderboard

| # | Experiment | Question | Headline | Scoring |
|---|-----------|----------|----------|---------|
| [001](001-titration-molmo2-vs-qwen/REPORT.md) | Molmo2 vs Qwen2.5-VL on titration | Does native-video grounding beat frames-as-images on timestamps? | VLMs are reliable step *sequencers*, unreliable *clocks*. Qwen's coverage collapse was later traced to an fps artifact — see 002. | manual |
| [002](002-qwen-fps-sweep/REPORT.md) | Qwen2.5-VL fps sweep (corrected) | Was 001's Qwen coverage collapse a model limit or a sampling artifact? | **Artifact.** Coverage scales with fps (60s→200s→325s); at fps≈0.5 Qwen2.5-VL-7B covers the full clip with correct ordering + repeats. Boundaries still soft. | manual |
| [003](003-cosmos-timestamped/REPORT.md) | Cosmos-Reason2-2B w/ on-frame timestamps | Does Cosmos ground when given its required input (timestamps drawn on frames)? | **Yes.** Bare frames loop/placeholder; with a Pillow `t=Ns` overlay it covers 0–330s with valid JSON. 2B semantics trail Qwen-7B — fair fight is Cosmos-8B (no MLX build; Spark target). | manual |

## Conventions

- **One folder per experiment**: `NNN-short-slug/`.
- **`config.json`** — every knob (clip, models, sampling, prompt, max_tokens, hardware). The reproducibility record.
- **`run.py`** — self-contained; regenerates `results/` from `config.json`. `python run.py` runs all cells; `python run.py <cell-id> ...` re-runs a subset.
- **`results/<cell>.json`** — raw per-cell output (committed; it's the evidence). Includes `degenerate: true` for failed cells — negative results are kept, not deleted.
- **`REPORT.md`** — the card: Hypothesis · Setup · Results · Findings · Verdict · Open threads.
- **Clips stay gitignored** (`clips/`), never committed; `config.json` records the source URL so a clip can be re-fetched.

## Adding an experiment

Copy `001-titration-molmo2-vs-qwen/`, bump the number, edit `config.json`, adjust `run.py` if the cells need a new input path, run it, then write `REPORT.md` and add a leaderboard row. (A capture skill to automate this scaffold is a candidate once the shape settles.)
