# Experiment 003 — Cosmos-Reason2-2B: overlay workaround vs. NVIDIA's official recipe

**Date**: 2026-06-22 · **Hardware**: Apple Silicon (MLX), mlx-vlm 0.6.3 · **Model**: `hzang/Cosmos-Reason2-2B-8bit` · **Scoring**: manual (LLM-judge harness TBD)

## Hypothesis

Cosmos-Reason2 (NVIDIA, Qwen3-VL base, reasoning-first) is the bake-off opponent. Does it run locally, how does NVIDIA actually intend it to be driven, and does it ground a 330s procedure on the Mac? Initial assumption (from the model card phrase "recognizes timestamps added at the bottom of each frame") was that we must burn timestamps into the frames. **That assumption was wrong** — see Findings.

## Setup

`hzang/Cosmos-Reason2-2B-8bit` is an MLX build (`model_type: qwen3_vl`) that loads in mlx-vlm with no conversion. We compare two methods (reproduce with `python run.py` / `python run.py official`; see `config.json`):

- **overlay** — sample frames with ffmpeg, draw `t=Ns` bottom-left with **Pillow** (this ffmpeg lacks `drawtext`), feed frames-as-images, parse `[{step,start,end}]` (brace-repair fallback for the 2B). `every_s=15` → 22 frames.
- **official** — NVIDIA's `temporal_localization.yaml` recipe verbatim: native `--video`, prompt *"Describe the notable events… json with 'mm:ss.ff'… 'start','end','caption'"*, reasoning OFF, `fps=0.5` (their fps=4 → ~1320 frames, infeasible on Mac).

Reasoning is **prompt-triggered**, not a flag: NVIDIA appends `REASONING_PROMPT` ("Answer using <think>…</think>, final answer after </think>") to the user turn. No `enable_thinking` exists.

## Results

| Method | Coverage | Outcome |
|--------|----------|---------|
| Bare frames, JSON-only (fps 0.2) | — | repetition loop |
| Bare frames, reasoning-prose prompt (fps 0.1) | — | fluent description, placeholder `0–330` single step |
| **Overlay** + refined prompt | **0–330s** | 10 steps, valid JSON, descriptive labels (tail collapses to "Record titre values" ×5) |
| **Official NVIDIA recipe** (native video, no overlay) | **0–4s** | **degenerate** — timestamps collapse to ~1s, captions loop, 8-bit quant bleeds Chinese ("试剂") |
| **Reasoning mode** (NVIDIA `REASONING_PROMPT`, temp 0.6) | n/a | real `<think>` trace, fluent order-of-operations CoT — but **missed the results table**, said the end "checks the worksheet" |
| Focused prose, no JSON, no reasoning | n/a | **best understanding** — separates procedure (0–~250s) from a **results/data table (300s+)** and reads the titre values (23.80 cm³ ×3, avg 23.80) |

Raw outputs in `results/`. All runs ~5–6GB peak, ~20–40s.

## Findings

1. **NVIDIA's native recipe FAILS on this MLX path.** Their method (native video + fps, no pixel overlay) relies on the Qwen3-VL processor passing frame-time metadata. On mlx-vlm's `qwen3_vl` route that metadata doesn't reach Cosmos — timestamps collapse to ~1s and captions loop. (The same native path worked for Qwen2.5-VL's `qwen2_5_vl` in 002, so this is a `qwen3_vl`-on-mlx gap, not a model failure.) **So the Pillow overlay is a *necessary local workaround*, not a detour** — it's the only way Cosmos got a real temporal anchor on the Mac. The official method needs vLLM/Spark.
2. **The "burn timestamps into frames" assumption was a misread.** NVIDIA's `vision.py` has no overlay code; "timestamps at the bottom of each frame" describes their *training data*. At inference they use processor frame-time metadata — which just happens to be broken on our MLX path, making the overlay incidentally correct.
3. **Reasoning is available but doesn't help the 2B.** The `REASONING_PROMPT` reliably produces a `<think>` trace, and the CoT is fluent and domain-aware — but it's *fluent-not-accurate*: it missed the results table the focused prose run read correctly. CoT ≠ better grounding at 2B.
4. **The JSON-only contract hid Cosmos's real strength.** In free prose it separates procedure from the results/data section and OCRs the titre values — the order-of-operations / structure understanding we actually care about. Forced into `[{step,start,end}]` that collapses to repeated labels.
5. **Runs light locally** (~5.6GB, no conversion) but NVIDIA validated **BF16 only**; the 8-bit quant visibly degrades (Chinese-character bleed, weaker CoT).

## Verdict

Cosmos-Reason2-2B **runs locally but no path yields a clean, trustworthy timeline on MLX**: the official recipe collapses, the overlay workaround grounds but with weak 2B semantics and a mushy repeat-tail, and reasoning mode is fluent-not-accurate. The honest Cosmos evaluation needs the **8B on the Spark via NVIDIA's `cosmos-reason2-inference`** (where the native recipe works). Local 2B/MLX is a feasibility probe, not the bake-off verdict. The most interesting signal: Cosmos's structure-aware prose ("this is procedure, that is a results table reading 23.80 cm³") is exactly the temporal-reasoning output the project wants — and the JSON timeline contract is the wrong container for it.

## Open threads

- [ ] **Cosmos-8B on the Spark** via NVIDIA's native pipeline (`temporal_localization.yaml`, fps=4, vLLM) — the recipe that actually works, at the size that actually reasons.
- [ ] **Why does mlx-vlm `qwen3_vl` drop frame-time metadata** (Cosmos collapses) when `qwen2_5_vl` carried it (Qwen2.5-VL grounded)? Worth an upstream look — would remove the overlay crutch.
- [ ] **Prose-first contract for reasoning models**: let Cosmos narrate (it catches the results section), then *parse* the timeline out of the narration, instead of demanding JSON. Possibly the right contract for the whole reasoning-model class.
- [ ] The LLM-judge scalar (carried) — and verify the 23.80 cm³ titre reading against the actual frames.
