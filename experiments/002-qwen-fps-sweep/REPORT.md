# Experiment 002 — Qwen2.5-VL fps sweep (corrected harness)

**Date**: 2026-06-22 · **Hardware**: Apple Silicon (MLX), mlx-vlm 0.6.3 · **Scoring**: manual (LLM-judge harness TBD)

## Hypothesis

Experiment 001 found Qwen2.5-VL-7B compressed a 330s titration into the first 10–80s and concluded its coverage was unreliable. Was that a model limitation, or an artifact of our harness? Qwen2.5-VL derives absolute timestamps from the fps handed to the processor — 001 ran at fps 0.0667, ~30× below the model's ~2 fps trained regime. If the collapse is an fps artifact, coverage should expand monotonically as we raise fps.

## Setup

Single model, single clip, single prompt — **only fps varies**. Path: `mlx_vlm.generate --video` (the qwen-vl-utils `video_generate` path is beta and collapses the clip to one frame here). Reproduce: `python run.py` (or `python run.py 0.5` for one point). See `config.json`.

```
second_per_grid_t = (1 / sample_fps) * temporal_patch_size   (temporal_patch_size = 2)
temporal position-ID step = second_per_grid_t * tokens_per_second   (tokens_per_second = 2)
```
Wrong fps → wrong seconds-per-ID → linearly mis-scaled (compressed) timestamps. Stating "330 seconds" in the prompt is only a sanity anchor; fps drives the math.

## Results

| fps | ~frames | coverage | steps | runtime |
|-----|---------|----------|-------|---------|
| 0.0667 | ~22 | 10–60s | 10 | 48s |
| 0.2 | ~66 | 10–200s | 10 | 54s |
| 0.5 | ~165 | **5–325s** | 19 | 75s |

Raw outputs in `results/`. At fps 0.5 the timeline spans the whole clip, recovers the **triplicate titration repeat** (fill burette → drip into flask → swirl → note reading, ×3), and reads the video's on-screen captions verbatim ("Don't forget to swirl flask", "Note down rough reading").

## Findings

1. **The 001 collapse was an fps artifact.** Coverage scales monotonically with fps (60s → 200s → 325s), exactly as `second_per_grid_t` predicts. Not a model limitation.
2. **fps is a first-class knob, not a cost dial.** It sets the time axis. Sample near the model's regime; too-sparse sampling silently compresses the timeline rather than just lowering resolution.
3. **Corrected, Qwen2.5-VL-7B is strong** on coverage + ordering + step semantics (it even OCRs the instructional captions). On this clip it reads richer than Molmo2's frames-as-images output in 001.
4. **Boundaries are still evenly-spaced estimates** (~5s spans, ~20s gaps). Coverage and ordering are now trustworthy; exact boundaries are not — unchanged from 001, consistent with the text-numeral output-paradigm ceiling. fps fixes *where* the timeline sits, not boundary precision.
5. **The "proper" path is broken here.** `mlx_vlm.video_generate` returned an empty array (128 frames) or saw a single "bottle" image (32 frames). It's flagged beta. Use `generate --video`.

## Verdict

**Revises 001.** Qwen2.5-VL-7B is the stronger candidate once fed fps in its trained regime — full coverage, correct ordering, faithful semantics. Boundary precision is still a separate, unsolved problem (output paradigm), not a model-choice problem. Practical recipe on Mac: `generate --video` at **fps ≈ 0.5** for a ~5-min clip; raise toward 1–2 (with a per-frame pixel budget) on the Spark.

## Open threads

- [ ] Push fps → 1–2 (frame/pixel-budget managed, or on the Spark): does boundary *spacing* tighten, or only coverage hold?
- [ ] `video_generate` beta breakage — investigate / file upstream, or just rely on vLLM video on the Spark.
- [ ] Re-run Molmo2 through an analogous native-video path (Spark/vLLM) for a fair head-to-head — 001 compared Molmo2-frames vs Qwen-video, an unfair axis.
- [ ] Define the LLM-judge scalar (carried from 001) so the next sweep auto-scores instead of eyeballing coverage.
