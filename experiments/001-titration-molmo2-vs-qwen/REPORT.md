# Experiment 001 — Molmo2 vs Qwen2.5-VL on a titration procedure

**Date**: 2026-06-22 · **Hardware**: Apple Silicon (MLX), mlx-vlm 0.6.3 · **Scoring**: manual (LLM-judge harness TBD)

## Hypothesis

Does a model *designed* for native video temporal grounding (Qwen2.5-VL: absolute-time encoding, dynamic FPS) produce sharper step timestamps than Molmo2 fed through the repo's frames-as-images workaround? And does either give timestamps we can trust for order-of-operations?

## Setup

Same clip, same output contract (`[{step, start, end}]`), one knob varied — model + input path. See `config.json`; reproduce with `python run.py` (or `python run.py <cell-id>` for one cell).

| Cell | Model | Input path | Sampling |
|------|-------|-----------|----------|
| `molmo2-8b-4bit` | Molmo2-8B-4bit | frames-as-images (`ground.py`) | 1 frame / 15s |
| `qwen2.5-vl-7b` | Qwen2.5-VL-7B-Instruct-bf16 | **native video** (`--video`) | fps 0.0667 |
| `qwen2.5-vl-3b` | Qwen2.5-VL-3B-Instruct-bf16 | **native video** (`--video`) | fps 0.0667 |

- Clip: FuseSchool titration, 330s ([source](https://www.youtube.com/watch?v=dZWxldXPIX8)). Gitignored; not redistributed.
- Why the input paths differ: Molmo2's mlx-vlm `--video` path is broken (replies "I can't view videos"), so it gets frames-as-images with injected timestamp labels. Qwen2.5-VL's native video path works. That asymmetry *is* the comparison.

## Results

| | Molmo2-8B (frames) | Qwen2.5-VL-7B (video) | Qwen2.5-VL-3B (video) |
|---|---|---|---|
| Runtime | 130s | 53s | 44s |
| Steps emitted | 20 | 14 | — (looped) |
| Step ordering | ✓ correct | ✓ correct | — |
| Step semantics | good — missed burette prep | **better** — caught rinse-water → rinse-acid → fill | — |
| Repeat structure | ✓ caught triplicate (rough + 2) | partial — one "repeat titration" | — |
| Boundaries | snap to the 15s sampling grid | own 5s guesses, **not** gridded | 1s increments, repeated |
| Temporal coverage | ✓ full 0–315s | ✗ collapsed to 10–80s of 330s | ✗ stuck ~19–28s |
| Outcome | usable timeline | rich but compressed | **degenerate** (repetition loop to token cap) |

Raw outputs in `results/`. Qwen-3B is kept as a negative result: small VLMs collapse into repetition on long structured output — **7B is the floor for this task**.

## Findings

1. **Ordering is the reliable signal; boundaries are not.** Both working models recover the correct step *sequence*. Neither produces trustworthy *timestamps* — Molmo2 snaps to the frame grid we injected; Qwen invents its own evenly-spaced guesses and compresses the whole procedure into the first 80s.
2. **Changing the model and the input path changed the failure mode, not the failure.** Grid-snap → timeline-compression. This is strong evidence the lever is the **output paradigm** (text-numeral generation is the weak one; continuous temporal decoding wins — see `@jin-2026-video-llm-time`, arXiv 2604.08966), not the backbone or native vs. sampled input.
3. **Qwen2.5-VL-7B has stronger procedure priors.** It surfaced the real titration prep (rinse burette with water, then acid, then fill) that Molmo2 skipped — a point in its favor for *semantic* step extraction, independent of timing.
4. **Native video works in mlx-vlm for Qwen2.5-VL** (5,186 prompt tokens = frames genuinely tokenized), unlike Molmo2. But it did not fix boundaries — see finding 2.

## Verdict

For procedure grounding today, treat the VLM as a **reliable step *sequencer* and an unreliable *clock*.** Don't pick a model on timestamp accuracy — none of these earn that trust through stock text-numeral output. Pick on step semantics (Qwen2.5-VL-7B leads here) and lean on ordering. Precise boundaries are a separate problem (continuous-decoding output paradigm; or a validation layer over the intervals).

## Open threads

- [ ] **Fair Qwen re-run**: its native temporal-grounding prompt format + denser fps — is the 10–80s coverage collapse a prompt artifact or real?
- [ ] **The judge**: define the LLM-as-judge scalar (ordering / coverage / boundary plausibility → one number) so 002+ auto-score instead of hand analysis.
- [ ] **Molmo2 on the Spark**: native video + real frame metadata via vLLM — does its boundary behavior change off the frames-as-images crutch?
- [ ] Add a ground-truth timeline for this clip to anchor any future numeric metric.
