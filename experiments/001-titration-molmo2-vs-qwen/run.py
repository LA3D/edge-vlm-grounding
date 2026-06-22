"""Experiment 001 — Molmo2 vs Qwen2.5-VL on a titration procedure clip.

Reproduces three cells on the same clip and writes results/<id>.json:
  - molmo2-8b-4bit : frames-as-images, via the repo's ground.py
  - qwen2.5-vl-7b  : native video, via `mlx_vlm.generate --video`
  - qwen2.5-vl-3b  : native video — kept as a negative result (repetition collapse)

The two paths differ on purpose: Molmo2's mlx-vlm video path is broken (see repo
README), so it gets frames-as-images with injected timestamp labels; Qwen2.5-VL has
a working native video path with absolute-time encoding. That difference *is* the
experiment. Same clip, same contract, one knob — model + input path — varied per cell.

Run from anywhere:  python experiments/001-titration-molmo2-vs-qwen/run.py
"""
import json, os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = json.load(open(os.path.join(HERE, "config.json")))
OUT = os.path.join(HERE, "results")


def salvage(text):  # pull the JSON array out of model chatter; None if no valid array
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def run_frames(cell, clip):  # Molmo2 via ground.py (in-process)
    sys.path.insert(0, ROOT)
    import ground
    return {"output": ground.ground(clip, every=cell["every_s"]), "raw": None}


def run_video(cell, clip):  # Qwen2.5-VL via the mlx-vlm CLI native video path
    cmd = [sys.executable, "-m", "mlx_vlm.generate", "--model", cell["model"],
           "--video", clip, "--fps", str(cell["fps"]),
           "--max-tokens", str(CFG["max_tokens"]), "--prompt", CFG["video_prompt"]]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    gen = r.stdout.split(CFG["video_prompt"])[-1]  # the CLI echoes the prompt (with an
    return {"output": salvage(gen), "raw": gen}    # example array) — keep only what follows


def main():
    os.makedirs(OUT, exist_ok=True)
    clip = os.path.join(ROOT, CFG["clip"])
    if not os.path.exists(clip):
        sys.exit(f"missing clip: {clip} — see repo README 'Getting a sample clip'")
    only = set(sys.argv[1:])  # optional: re-run just the named cells
    for cell in CFG["cells"]:
        if only and cell["id"] not in only:
            continue
        t = time.time()
        runner = run_frames if cell["path"] == "frames" else run_video
        res = runner(cell, clip)
        res.update(cell=cell["id"], model=cell["model"], path=cell["path"],
                   fps=cell.get("fps"), every_s=cell.get("every_s"),
                   seconds=round(time.time() - t, 1),
                   n_steps=len(res["output"]) if res["output"] else 0,
                   degenerate=res["output"] is None)
        json.dump(res, open(os.path.join(OUT, cell["id"] + ".json"), "w"), indent=2)
        tag = "DEGENERATE" if res["degenerate"] else f"{res['n_steps']} steps"
        print(f"{cell['id']:22s} {tag:12s} {res['seconds']}s")


if __name__ == "__main__":
    main()
