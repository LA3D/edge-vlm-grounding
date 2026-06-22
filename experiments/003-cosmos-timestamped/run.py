"""Experiment 003 — Cosmos-Reason2-2B with timestamps burned into frames.

Cosmos-Reason2 localizes by reading timestamps drawn on the frames (NVIDIA docs:
"recognizes timestamps added at the bottom of each frame"), not from fps encoding.
Bare frames (001/002 style) gave it no temporal anchor — it looped or returned a
placeholder `[{"step":"<label>","start":0,"end":330}]`. Here we sample frames with
ffmpeg, draw "t=Ns" on each with Pillow (this ffmpeg build lacks drawtext), and feed
the stamped sequence; Cosmos reasons, then emits the ordered timeline.

Run:  python experiments/003-cosmos-timestamped/run.py
"""
import json, os, re, sys, glob, tempfile, subprocess, time
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = json.load(open(os.path.join(HERE, "config.json")))
OUT = os.path.join(HERE, "results")


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


def parse(text):  # strip the reasoning, take the last JSON array (repairing a common malformation)
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


def main():
    os.makedirs(OUT, exist_ok=True)
    clip = os.path.join(ROOT, CFG["clip"])
    if not os.path.exists(clip):
        sys.exit(f"missing clip: {clip} — see repo README 'Getting a sample clip'")
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        frames = sample_and_stamp(clip, CFG["every_s"], tmp)
        model, processor = load(CFG["model"])
        fmt = apply_chat_template(processor, model.config, CFG["prompt"], num_images=len(frames))
        out = generate(model, processor, fmt, image=frames, max_tokens=CFG["max_tokens"], verbose=False)
    text = getattr(out, "text", out)
    steps = parse(text)
    res = {"model": CFG["model"], "every_s": CFG["every_s"], "n_frames": len(frames),
           "output": steps, "n_steps": len(steps) if steps else 0,
           "coverage": coverage(steps), "seconds": round(time.time() - t, 1),
           "degenerate": steps is None, "raw": text}
    json.dump(res, open(os.path.join(OUT, "cosmos-2b.json"), "w"), indent=2)
    cov = res["coverage"]
    covs = f"{cov[0]:.0f}-{cov[1]:.0f}s" if cov else "—"
    print(f"cosmos-2b  {res['n_steps']} steps  cover {covs}  {res['seconds']}s  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
