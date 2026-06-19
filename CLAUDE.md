# CLAUDE.md — edge-vlm-grounding

Project context for Claude Code sessions in this repo.

## What this is

Back-of-the-envelope experiments: small edge VLMs (Molmo2 first) for **ordered, timestamped temporal grounding of experimental procedures** in video. Dev on Mac (MLX) → deploy on the DGX Spark (vLLM/NVFP4). Eventual Molmo2-vs-Cosmos-Reason2 bake-off.

## Source of truth (Obsidian vault)

- **Full plan**: `~/Obsidian/obsidian/01 - Projects/Edge VLM Procedure Grounding/PLAN.md`
- **Project note**: `~/Obsidian/obsidian/01 - Projects/Edge VLM Procedure Grounding.md`
- **Concept / landscape**: `~/Obsidian/obsidian/03 - Resources/Video Temporal Grounding.md`

Planning, phases, and decisions live in the vault — this repo mirrors the runnable parts. Update the vault PLAN when scope changes.

## Conventions

- Python via `uv` (venv at `.venv/`, Python 3.12).
- Inference via `mlx-vlm` (Mac) / vLLM (Spark). Keep the calling code OpenAI-API-shaped so Mac and Spark are swappable by base URL.
- The grounding output contract is `[{"step", "start", "end"}, ...]` in temporal order. Don't change it without updating the vault PLAN (Phase 2 scoring depends on it).

## Key files

- `ground.py` — first grounding script: video → strict JSON step timeline.
- `README.md` — quickstart (Phase 0–1).

## Stakeholders

Chuck Vardeman (lead), Andrey Kuehlkamp (CRC, runs the Spark), Paul Brenner (CRC).
