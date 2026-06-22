# edge-vlm-grounding

Back-of-the-envelope experiments using small, edge-deployable vision-language models to **temporally ground experimental procedures** in video — recovering the *ordered, timestamped* sequence of steps (not single-moment retrieval).

Anchor model: **Molmo2** (Ai2), with **Qwen2.5-VL** (Alibaba) and **Cosmos-Reason2** (NVIDIA) in a local MLX comparison ahead of a bake-off on the NVIDIA DGX Spark (GB10 Grace Blackwell) via vLLM/NVFP4. The calling code is kept OpenAI-API-shaped so Mac and Spark are swappable by base URL.

Full plan, phases, and research context live in the Obsidian vault (see `CLAUDE.md`).

## Why

A lab DGX Spark was crashing under Ollama on larger models — likely a unified-memory mismatch (llama.cpp treats GPU/host memory as separate pools; the GB10 is one 128 GB unified pool). Moving to vLLM is the runtime fix. On top of it we want a VLM that watches a procedure video and emits the ordered step timeline.

## Model: Molmo2 (Ai2)

A family of open-weights **and** open-data VLMs from the Allen Institute for AI, built for image, multi-image, and video understanding **with grounding** — which is why it's the starting point here. Molmo2-8B pairs a Qwen3-8B language model with a SigLIP 2 vision backbone, and the same architecture serves on both MLX and vLLM.

| | |
|---|---|
| Paper | [Molmo2: Open Weights and Data for VLMs with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611) (arXiv 2601.10611) · [tech report PDF](https://www.datocms-assets.com/64837/1766008501-molmo2-tech-report.pdf) |
| Announcement | [Molmo 2: SOTA video understanding, pointing, and tracking](https://allenai.org/blog/molmo2) |
| Model card | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) (also [Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B)) |
| MLX build | [mlx-community/Molmo2-8B-4bit](https://huggingface.co/mlx-community/Molmo2-8B-4bit) — converted from `allenai/Molmo2-8B` |
| Code | [github.com/allenai/molmo2](https://github.com/allenai/molmo2) · [Ai2 platform docs](https://docs.allenai.org/models/molmo2) |

## Also under test (local MLX comparison)

The bake-off isn't Molmo2-only — we're comparing three open VLMs locally before the Spark. The load-bearing lesson from the experiments: **these are not interchangeable "feed it frames" boxes. Each takes its temporal signal through a different channel, and missing it silently degrades grounding** (timeline compression, repetition loops, placeholder timelines — never a clean error).

| Model | How time enters | Local MLX status |
|-------|-----------------|------------------|
| **Molmo2-8B** (Ai2) | textual frame labels we inject (`image N: t=Xs`) | works via frames-as-images; boundaries snap to the label grid |
| **Qwen2.5-VL-7B** (Alibaba) | native `fps` → absolute-time encoding (mRoPE) | strongest local result once `fps` is in its trained regime — [exp 002](experiments/002-qwen-fps-sweep/REPORT.md) |
| **Cosmos-Reason2-2B** (NVIDIA) | pixel timestamps drawn on the frame | native recipe collapses on mlx-vlm `qwen3_vl`; needs a drawn `t=Ns` overlay — [exp 003](experiments/003-cosmos-timestamped/REPORT.md) |

## Grounding contract

`ground.py` forces the output contract used for scoring, identical on Mac and Spark:

```json
[{"step": "<short label>", "start": <seconds>, "end": <seconds>}, ...]
```

Steps are in temporal order. Don't change the schema without updating the vault PLAN — Phase 2 scoring depends on it.

## Quickstart (Mac / MLX)

> `mlx-vlm` flags move fast. If a command errors, run `python -m mlx_vlm.generate --help`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx-vlm

# pull a Molmo2 MLX build (lightest first)
huggingface-cli download mlx-community/Molmo2-8B-4bit
# or a smaller 4B build: huggingface-cli download Aliyovic/molmo2-4b-mlx-8bit

# ordered, timestamped grounding -> strict JSON
python ground.py path/to/sample_procedure.mp4 [every_seconds]   # default: 1 frame / 6s
```

`ground.py` loads the model in-process via `mlx-vlm`. For the portable, base-URL-swappable path see [Running on the DGX Spark](#running-on-the-dgx-spark-vllm) below.

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
- `ytsearch10:<query>` searches instead of needing a URL; `--match-filter` keeps it short. For a real lab procedure, swap the query, e.g. `"ytsearch10:acid base titration procedure step by step"`.
- These clips are transient local research inputs, not redistributed.

### Gotcha: native `--video` doesn't ground (mlx-vlm 0.6.3)

`mlx_vlm.generate --video ...` accepts the flag but **does not inject frames** for Molmo2 — the model replies *"I can't view videos."* The single-image path works fine, so `ground.py` samples frames with `ffmpeg` (`fps=1/every`) and feeds them as a **timestamped multi-image sequence**. Requires `ffmpeg` on PATH. This also mirrors how the vLLM/Spark side takes sampled frames. Tune `every_seconds` to trade timestamp resolution against frame count / memory — coarse intervals (12–15s) keep the frame count sane on longer clips.

### Gotcha: text-numeral timestamps are the low-fidelity paradigm

`ground.py` asks the model to emit `start`/`end` as **numbers inside JSON** — text-numeral generation. A 2026 controlled study across compact VLMs ([How Should Video LLMs Output Time?](https://arxiv.org/abs/2604.08966)) found this is the *weakest* output paradigm: ~27.5 mIoU on Charades-STA, vs. 57.1 for continuous temporal decoding at the same 8B scale. Combined with the frames-as-images workaround (which feeds the sampling grid back to the model as text labels), expect **step ordering to be reliable but boundaries to snap to the sampling interval** — confirmed on a real titration clip: correct step order and repeat structure, timestamps quantized to the 15s grid.

This is fine for validating the prompt and output shape; it is *not* a boundary-accuracy claim. Precise timestamps come from the Spark path (native video metadata + continuous decoding). Order-of-operations can be enforced independently by validating the emitted intervals against protocol constraints — Allen interval relations / SHACL shapes — so soft boundaries don't break sequence guarantees.

## Running on the DGX Spark (vLLM)

Molmo2 is supported in **vLLM ≥ 0.15.0**. Install the vision preprocessing helper (`molmo-utils`, modeled on `qwen-vl-utils`) and serve the OpenAI-compatible HTTP endpoint:

```bash
uv pip install "vllm>=0.15.0" molmo-utils

# NVFP4 weights for the GB10; swap in the bf16 card if you want a fidelity baseline
vllm serve allenai/Molmo2-8B --port 8080
```

vLLM exposes an **OpenAI Vision–compatible** API, so the same client code drives Mac and Spark — only the base URL changes:

```python
from openai import OpenAI

# Mac:   python -m mlx_vlm.server --model mlx-community/Molmo2-8B-4bit --port 8080
# Spark: vllm serve allenai/Molmo2-8B --port 8080
client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="allenai/Molmo2-8B",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},                                  # the grounding prompt
        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
          for b64 in frames_b64],                                          # sampled, timestamped frames
    ]}],
    max_tokens=1024,
)
```

The frame-sampling + prompt logic in `ground.py` is the same on both sides; only the transport (in-process `mlx-vlm` vs. HTTP) differs.

## Experiments

Structured, reproducible logs live in [`experiments/`](experiments/) — one folder per run (captured config, re-runnable `run.py`, raw results, `REPORT.md` card; [autoresearch](https://github.com/karpathy/autoresearch)-style, minus the autonomous loop for now). Start at [`experiments/README.md`](experiments/README.md).

| # | What | Headline |
|---|------|----------|
| [001](experiments/001-titration-molmo2-vs-qwen/REPORT.md) | Molmo2 vs Qwen2.5-VL on titration | VLMs are reliable step *sequencers*, unreliable *clocks* |
| [002](experiments/002-qwen-fps-sweep/REPORT.md) | Qwen2.5-VL fps sweep | coverage scales with `fps`; 001's "collapse" was 30× too-sparse sampling |
| [003](experiments/003-cosmos-timestamped/REPORT.md) | Cosmos-Reason2-2B | NVIDIA's native recipe collapses on MLX; a `t=Ns` overlay is the local workaround |

**The through-line:** every model gets *ordering* right and *boundary precision* wrong, in a model-specific way. Treat the VLM as a reliable step **sequencer** and an unreliable **clock** — enforce order-of-operations separately (validate the emitted intervals against protocol constraints: Allen interval relations / SHACL), rather than trusting raw timestamps.

## Caveat

MLX 4-bit ≠ Spark NVFP4. Treat the Mac as a *functional* prototype (does the prompt work, is the output shape right), not a fidelity/throughput proxy for the Spark. Re-validate accuracy on the Spark before drawing bake-off conclusions.

## License

MIT — see `LICENSE`.
