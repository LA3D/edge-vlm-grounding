"""Experiment 003 — Cosmos-Reason2-2B: overlay workaround vs. NVIDIA's official recipe.

Cosmos-Reason2 is documented to localize from native video at fps=4 with a
`{start, end, caption}` JSON prompt (reasoning OFF) — NVIDIA does NOT burn
timestamps into pixels at inference (that's a training-data convention). But on
this mlx-vlm `qwen3_vl` path the native recipe fails: timestamps collapse to ~1s
and captions loop. So two modes here:

  overlay  (default): sample frames, draw "t=Ns" with Pillow, feed frames-as-images.
                      A workaround that gives Cosmos a temporal anchor the MLX path
                      otherwise withholds. -> full 0-330s coverage.
  official ('official' arg): native --video + NVIDIA's temporal_localization prompt,
                      no overlay, no reasoning. -> degenerate here (timestamp collapse).

Run:  python experiments/003-cosmos-timestamped/run.py            # overlay
      python experiments/003-cosmos-timestamped/run.py official   # NVIDIA recipe
"""
import json, os, re, sys, glob, tempfile, subprocess, time
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = json.load(open(os.path.join(HERE, "config.json")))
OUT = os.path.join(HERE, "results")


# ---------- overlay mode ----------

def sample_and_stamp(clip, every, tmp):  # -> [stamped jpg paths], one frame per `every` seconds
    subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vf", f"fps=1/{every}",
                    os.path.join(tmp, "f_%04d.jpg"), "-y"], check=True)
    frames = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
    font = ImageFont.truetype(CFG["font"], CFG["fontsize"])
    fs, stamped = CFG["fontsize"], []
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB")
        d = ImageDraw.Draw(im)
        x, y = 20, im.height - fs - 24
        d.rectangle([x - 8, y - 6, x + fs * 4, y + fs + 10], fill="black")  # contrast box
        d.text((x, y), f"t={i * every}s", fill="white", font=font)
        p = os.path.join(tmp, f"s_{i:04d}.jpg")
        im.save(p)
        stamped.append(p)
    return stamped


def _repair(b):  # the 2B sometimes drops the opening brace on each object
    b = re.sub(r'},\s*"step"', '}, {"step"', b)
    b = re.sub(r'\[\s*"step"', '[{"step"', b)
    return b


def parse_steps(text):  # strip reasoning, take the last JSON array (repairing a common malformation)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    s, e = text.rfind("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    blob = text[s:e + 1]
    for attempt in (blob, _repair(blob)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def coverage(steps):
    if not steps:
        return None
    return [min(s["start"] for s in steps), max(s["end"] for s in steps)]


def run_overlay(clip):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        frames = sample_and_stamp(clip, CFG["every_s"], tmp)
        model, processor = load(CFG["model"])
        fmt = apply_chat_template(processor, model.config, CFG["prompt"], num_images=len(frames))
        out = generate(model, processor, fmt, image=frames, max_tokens=CFG["max_tokens"], verbose=False)
    text = getattr(out, "text", out)
    steps = parse_steps(text)
    return {"mode": "overlay", "every_s": CFG["every_s"], "n_frames": len(frames),
            "output": steps, "n_steps": len(steps) if steps else 0,
            "coverage": coverage(steps), "seconds": round(time.time() - t, 1),
            "degenerate": steps is None, "raw": text}, "cosmos-2b.json"


# ---------- official mode (NVIDIA temporal_localization recipe) ----------

def mmssff_to_s(v):  # "mm:ss.ff" -> seconds
    try:
        m, rest = v.split(":")
        return int(m) * 60 + float(rest)
    except Exception:
        return None


def run_official(clip):
    cmd = [sys.executable, "-m", "mlx_vlm.generate", "--model", CFG["model"],
           "--video", clip, "--fps", str(CFG["official_fps"]),
           "--temperature", "0.7", "--max-tokens", str(CFG["max_tokens"]),
           "--prompt", CFG["official_prompt"]]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    gen = r.stdout.split(CFG["official_prompt"])[-1]
    times = [x for x in (mmssff_to_s(v) for v in re.findall(r'"(?:start|end)":\s*"([0-9:.]+)"', gen)) if x is not None]
    caps = re.findall(r'"caption":\s*"(.*?)"', gen)
    cov = [min(times), max(times)] if times else None
    uniq = len(set(caps))
    # degenerate if it never spreads past ~30s of a 330s clip, or mostly-duplicate captions
    degen = (not times) or (cov and cov[1] - cov[0] < 30) or (bool(caps) and uniq / len(caps) < 0.5)
    return {"mode": "official", "fps": CFG["official_fps"], "n_events": len(caps),
            "unique_captions": uniq, "coverage_s": cov, "seconds": round(time.time() - t, 1),
            "degenerate": bool(degen), "raw": gen.strip()[:4000]}, "cosmos-2b-official.json"


def main():
    os.makedirs(OUT, exist_ok=True)
    clip = os.path.join(ROOT, CFG["clip"])
    if not os.path.exists(clip):
        sys.exit(f"missing clip: {clip} — see repo README 'Getting a sample clip'")
    mode = sys.argv[1] if len(sys.argv) > 1 else "overlay"
    res, fname = run_official(clip) if mode == "official" else run_overlay(clip)
    json.dump(res, open(os.path.join(OUT, fname), "w"), indent=2)
    if mode == "official":
        cov = res["coverage_s"]
        covs = f"{cov[0]:.0f}-{cov[1]:.0f}s" if cov else "—"
        tag = "DEGENERATE" if res["degenerate"] else "ok"
        print(f"official  {res['n_events']} events ({res['unique_captions']} unique)  cover {covs}  {tag}  {res['seconds']}s")
    else:
        cov = res["coverage"]
        covs = f"{cov[0]:.0f}-{cov[1]:.0f}s" if cov else "—"
        print(f"overlay  {res['n_steps']} steps  cover {covs}  {res['seconds']}s  ({res['n_frames']} frames)")


if __name__ == "__main__":
    main()
