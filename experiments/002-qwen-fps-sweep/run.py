"""Experiment 002 — Qwen2.5-VL fps sweep (corrected harness).

Tests whether 001's Qwen coverage collapse (a 330s clip compressed into 10-80s)
was an artifact of too-sparse frame sampling. Qwen2.5-VL derives absolute
timestamps from the fps passed to the processor:

    second_per_grid_t = (1 / sample_fps) * temporal_patch_size   (temporal_patch_size=2)
    temporal position-ID step = second_per_grid_t * tokens_per_second   (tokens_per_second=2)

001 ran at fps 0.0667 — ~30x below the model's ~2fps trained regime — so the time
axis was badly mis-scaled. Here we sweep fps and watch coverage track it.

Path note: mlx_vlm.video_generate (the qwen-vl-utils port that should be "proper")
is beta and collapses the clip to a single frame on this setup, so we use the
generic `generate --video`, which does feed frames and respond to fps.

Run:  python experiments/002-qwen-fps-sweep/run.py [fps ...]   # default: the configured sweep
"""
import json, os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = json.load(open(os.path.join(HERE, "config.json")))
OUT = os.path.join(HERE, "results")


def salvage(text):
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def coverage(steps):  # (earliest start, latest end) — how much of the clip the timeline spans
    if not steps:
        return None
    return [min(s["start"] for s in steps), max(s["end"] for s in steps)]


def run_fps(fps, clip):
    cmd = [sys.executable, "-m", "mlx_vlm.generate", "--model", CFG["model"],
           "--video", clip, "--fps", str(fps),
           "--max-tokens", str(CFG["max_tokens"]), "--prompt", CFG["prompt"]]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    gen = r.stdout.split(CFG["prompt"])[-1]  # drop the echoed prompt (with its example array)
    out = salvage(gen)
    return {"fps": fps, "output": out, "n_steps": len(out) if out else 0,
            "coverage": coverage(out), "seconds": round(time.time() - t, 1),
            "degenerate": out is None}


def main():
    os.makedirs(OUT, exist_ok=True)
    clip = os.path.join(ROOT, CFG["clip"])
    if not os.path.exists(clip):
        sys.exit(f"missing clip: {clip} — see repo README 'Getting a sample clip'")
    sweep = [float(x) for x in sys.argv[1:]] or CFG["fps_sweep"]
    for fps in sweep:
        res = run_fps(fps, clip)
        json.dump(res, open(os.path.join(OUT, f"fps-{fps}.json"), "w"), indent=2)
        cov = res["coverage"]
        covs = f"{cov[0]:.0f}-{cov[1]:.0f}s" if cov else "—"
        print(f"fps-{fps:<7} {res['n_steps']:2d} steps  cover {covs:12s} {res['seconds']}s")


if __name__ == "__main__":
    main()
