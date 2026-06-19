# edge-vlm-grounding

Back-of-the-envelope experiments using small, edge-deployable vision-language models to **temporally ground experimental procedures** in video — recovering the *ordered, timestamped* sequence of steps (not single-moment retrieval).

Starting model: **Molmo2** (Ai2) — runs locally on Apple Silicon via MLX and on the NVIDIA DGX Spark (GB10 Grace Blackwell) via vLLM/NVFP4. Goal is a Molmo2-vs-Cosmos-Reason2 bake-off on the Spark.

Full plan, phases, and research context live in the Obsidian vault (see `CLAUDE.md`).

## Why

A lab DGX Spark was crashing under Ollama on larger models — likely a unified-memory mismatch (llama.cpp treats GPU/host memory as separate pools; the GB10 is one 128 GB unified pool). Moving to vLLM is the runtime fix. On top of it we want a VLM that watches a procedure video and emits the ordered step timeline.

## Quickstart (Mac / MLX)

> `mlx-vlm` flags move fast. If a command errors, run `python -m mlx_vlm.generate --help`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx-vlm

# pull a Molmo2 MLX build (lightest first)
huggingface-cli download Aliyovic/molmo2-4b-mlx-8bit
# or: huggingface-cli download mlx-community/Molmo2-8B-4bit

# ordered, timestamped grounding -> strict JSON
python ground.py path/to/sample_procedure.mp4 [every_seconds]   # default: 1 frame / 6s
```

### Getting a sample clip

No real procedure footage yet? Grab a short public stand-in into `clips/` (gitignored — clips are never committed). Needs `yt-dlp`, `ffmpeg`, **and a JavaScript runtime** — modern YouTube hands back a JS challenge that `yt-dlp` can only solve with one:

```bash
brew install deno          # JS runtime for yt-dlp's challenge solver
uv pip install yt-dlp      # ffmpeg should already be on PATH

mkdir -p clips
yt-dlp \
  --remote-components ejs:github \
  --match-filter "duration < 75 & duration > 20" \
  -f "mp4[height<=480]/best[height<=480]" \
  --no-playlist --max-downloads 1 \
  -o "clips/standin.%(ext)s" \
  "ytsearch10:how to make a quesadilla step by step"
```

- `--remote-components ejs:github` is the part that matters — it fetches the EJS solver `yt-dlp` runs under `deno` to clear YouTube's challenge. Without it (or without a JS runtime), extraction falls back to a degraded mode with missing formats.
- `ytsearch10:<query>` searches instead of needing a URL; `--match-filter` keeps it short.
- Swap the query / drop the `*.mp4` filter for real footage. These clips are transient local research inputs, not redistributed.

`ground.py` forces the output contract used for scoring:

```json
[{"step": "<short label>", "start": <seconds>, "end": <seconds>}, ...]
```

That schema carries over unchanged to the Spark.

### Gotcha: native `--video` doesn't ground (mlx-vlm 0.6.3)

`mlx_vlm.generate --video ...` accepts the flag but **does not inject frames** for Molmo2 — the model replies *"I can't view videos."* The single-image path works fine, so `ground.py` samples frames with `ffmpeg` (`fps=1/every`) and feeds them as a **timestamped multi-image sequence**. Requires `ffmpeg` on PATH. This also mirrors how the vLLM/Spark side will take sampled frames. Tune `every_seconds` to trade timestamp resolution against frame count / memory.

### Optional: OpenAI-compatible server (matches the Spark API)

```bash
python -m mlx_vlm.server --model mlx-community/Molmo2-8B-4bit --port 8080
```

Calling this with the OpenAI client means the same code works against vLLM on the Spark later — just swap the base URL.

## Caveat

MLX 4-bit ≠ Spark NVFP4. Treat the Mac as a *functional* prototype (does the prompt work, is the output shape right), not a fidelity/throughput proxy for the Spark.

## License

MIT — see `LICENSE`.
