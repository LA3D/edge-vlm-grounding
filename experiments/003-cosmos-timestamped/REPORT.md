# Experiment 003 — Cosmos-Reason2-2B with timestamps burned into frames

**Date**: 2026-06-22 · **Hardware**: Apple Silicon (MLX), mlx-vlm 0.6.3 · **Model**: `hzang/Cosmos-Reason2-2B-8bit` · **Scoring**: manual (LLM-judge harness TBD)

## Hypothesis

Cosmos-Reason2 (NVIDIA, Qwen3-VL base, reasoning-first) is documented to localize by reading **timestamps drawn on the frames** — not from fps encoding like Qwen2.5-VL. Fed bare frames it should fail to ground; fed frames with a visible `t=Ns` overlay it should produce a real timeline. Also: does it even run locally (it's pitched as a vLLM/Spark model)?

## Setup

`hzang/Cosmos-Reason2-2B-8bit` (an MLX build — `model_type: qwen3_vl`) loads in mlx-vlm with no conversion. Method: sample frames with ffmpeg, draw `t=Ns` bottom-left with **Pillow** (this ffmpeg build lacks `drawtext`), feed the stamped sequence frames-as-images, let it reason, parse the trailing JSON (with a brace-repair fallback — the 2B drops `{` on objects). `every_s=15` → 22 frames. Reproduce: `python run.py`. See `config.json`.

## Results

| Input | Coverage | Output |
|-------|----------|--------|
| Bare frames, JSON-only prompt (fps 0.2) | — | **repetition loop** ("Fill burette…" ×N to token cap) |
| Bare frames, reasoning prompt (fps 0.1) | — | fluent *description*, but placeholder `[{"step":"<label>","start":0,"end":330}]` |
| **Stamped frames**, generic prompt | 0–330s | full coverage, but "Step 1…22" labels + malformed JSON |
| **Stamped frames**, refined prompt + repair | **0–330s** | **10 steps, valid JSON, descriptive labels** (tail collapses to "Record titre values" ×5) |

Final timeline: prepare/clean burette → fill with acid → titrate (30–120s) → record titre → clean burette → record… Raw reasoning + output in `results/cosmos-2b.json`. ~18s, peak ~5GB.

## Findings

1. **The overlay is the unlock.** With timestamps drawn on the frames, Cosmos grounds to full 0–330s coverage; without them it loops or returns a placeholder. This is a *third* temporal-signal channel across our three models — Molmo2 (textual frame labels), Qwen2.5-VL (fps/absolute-time encoding), **Cosmos (pixel timestamps on the frame)**.
2. **It reads on-frame timestamps precisely.** An 8-frame probe: it reported "first image t=0s, last image t=140s" exactly. Strong OCR of overlays (it also reads the video's own captions).
3. **The 2B is fragile.** Needed prompt coaxing to merge frames into steps (vs one-per-frame), a JSON brace-repair fallback, and it still collapses the triplicate-repeat tail into repeated "Record titre values". Reasoning-first + 2B is a delicate combination on long structured output.
4. **It's genuinely edge-class on Mac** — ~5GB, ~18s, no conversion needed. Runs locally despite being pitched as a vLLM/Spark model.

## Verdict

Cosmos-Reason2 is **viable locally and grounds well once given its required input** (on-frame timestamps). But the 2B's output quality — label semantics, JSON validity, repeat-region collapse — **trails Qwen2.5-VL-7B (002)**. That's an unfair size axis (2B vs 7B); the honest Cosmos comparison is the **8B**, which has no MLX build yet and is really a Spark/vLLM target (NVIDIA ships `cosmos-reason2-inference` + a `temporal_localization.yaml`). Local 2B is a useful smoke test of the model family and the overlay harness, not the bake-off verdict.

## Open threads

- [ ] **Cosmos-8B**: convert to MLX, or run on the Spark via NVIDIA's native `cosmos-reason2-inference` + `temporal_localization.yaml` (its blessed temporal path) for a size-fair fight.
- [ ] **Cross-pollinate**: do on-frame timestamp overlays *also* sharpen Molmo2 / Qwen boundaries? The overlay is model-agnostic; worth an A/B on the leaders.
- [ ] Denser frames (`every_s` 5–10) — does the 2B's repeat-region collapse improve, or is it a capacity ceiling?
- [ ] The LLM-judge scalar (carried from 001/002) — three models now produce timelines; eyeballing doesn't scale.
