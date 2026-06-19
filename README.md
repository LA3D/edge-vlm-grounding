# edge-vlm-grounding

Back-of-the-envelope experiments using small, edge-deployable vision-language models to **temporally ground experimental procedures** in video — recovering the *ordered, timestamped* sequence of steps (not single-moment retrieval).

Starting model: **Molmo2** (Ai2) — runs locally on Apple Silicon via MLX and on the NVIDIA DGX Spark (GB10 Grace Blackwell) via vLLM/NVFP4. Goal is a Molmo2-vs-Cosmos-Reason2 bake-off on the Spark.

Full plan, phases, and research context live in the Obsidian vault (see `CLAUDE.md`).

## Why

A lab DGX Spark was crashing under Ollama on larger models — likely a unified-memory mismatch (llama.cpp treats GPU/host memory as separate pools; the GB10 is one 128 GB unified pool). Moving to vLLM is the runtime fix. On top of it we want a VLM that watches a procedure video and emits the ordered step timeline.

## Quickstart (Mac / MLX)

> `mlx-vlm` flags move fast. If a command errors, run `python -m mlx_vlm.generate --help`. Video support needs a recent `mlx-vlm`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx-vlm

# pull a Molmo2 MLX build (lightest first)
huggingface-cli download Aliyovic/molmo2-4b-mlx-8bit
# or: huggingface-cli download mlx-community/Molmo2-8B-4bit

# smoke test: video -> free text
python -m mlx_vlm.generate \
  --model mlx-community/Molmo2-8B-4bit \
  --video path/to/sample_procedure.mp4 \
  --prompt "Describe what happens in this video, step by step." \
  --max-tokens 512

# ordered, timestamped grounding -> strict JSON
python ground.py path/to/sample_procedure.mp4
```

`ground.py` forces the output contract used for scoring:

```json
[{"step": "<short label>", "start": <seconds>, "end": <seconds>}, ...]
```

Iterate the prompt until output reliably parses in temporal order. That schema carries over unchanged to the Spark.

### Optional: OpenAI-compatible server (matches the Spark API)

```bash
python -m mlx_vlm.server --model mlx-community/Molmo2-8B-4bit --port 8080
```

Calling this with the OpenAI client means the same code works against vLLM on the Spark later — just swap the base URL.

## Caveat

MLX 4-bit ≠ Spark NVFP4. Treat the Mac as a *functional* prototype (does the prompt work, is the output shape right), not a fidelity/throughput proxy for the Spark.

## License

MIT — see `LICENSE`.
