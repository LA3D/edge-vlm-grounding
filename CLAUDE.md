# CLAUDE.md — edge-vlm-grounding

Project context for Claude Code sessions in this repo.

## What this is

Back-of-the-envelope experiments: small edge VLMs for **ordered, timestamped temporal grounding of experimental procedures** in video. Local comparison on Mac (MLX) across Molmo2 (Ai2), Qwen2.5-VL (Alibaba), and Cosmos-Reason2 (NVIDIA) → deploy and bake-off on the DGX Spark (vLLM/NVFP4).

## Planning & findings

The runnable record and experiment findings live in this repo — see `experiments/` (one folder per run: captured config, re-runnable `run.py`, raw results, a `REPORT.md` card). Detailed planning is kept in the author's private notes, not in this repo.

## Conventions

- Python via `uv` (venv at `.venv/`, Python 3.12).
- Inference via `mlx-vlm` (Mac) / vLLM (Spark). Keep the calling code OpenAI-API-shaped so Mac and Spark are swappable by base URL.
- The grounding output contract is `[{"step", "start", "end"}, ...]` in temporal order. It's the scoring contract — don't change it lightly.

## Key files

- `ground.py` — first grounding script: video → strict JSON step timeline.
- `experiments/` — reproducible experiment logs (config + `run.py` + results + `REPORT.md` per run); start at `experiments/README.md`.
- `README.md` — quickstart and findings.

## Stakeholders

Chuck Vardeman (lead), Andrey Kuehlkamp (CRC, runs the Spark), Paul Brenner (CRC).
