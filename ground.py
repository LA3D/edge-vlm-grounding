"""Molmo2 (MLX) → ordered, timestamped step timeline for a procedure video.

mlx-vlm 0.6.3's native --video path does not inject frames for Molmo2 (the model
replies "I can't view videos"). Workaround that actually grounds: sample frames at
a fixed interval with ffmpeg and feed them as a timestamped multi-image sequence.
This also matches how the vLLM/Spark side will take sampled frames.

Usage:
    python ground.py path/to/procedure.mp4 [every_seconds]

Output: JSON array of {"step", "start", "end"} in temporal order.
"""
import json, sys, os, glob, tempfile, subprocess

MODEL = "mlx-community/Molmo2-8B-4bit"


def duration(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", video],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def sample_frames(video, every, tmp):  # -> [(timestamp_sec, jpg_path), ...]
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps=1/{every}",
                    os.path.join(tmp, "f_%04d.jpg"), "-y"], check=True)
    frames = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
    return [(i * every, f) for i, f in enumerate(frames)]


def build_prompt(stamps, total):
    idx = "\n".join(f"- image {i + 1}: t={t:.0f}s" for i, (t, _) in enumerate(stamps))
    return (
        f"These {len(stamps)} images are frames sampled in order from a {total:.0f}s "
        f"video of a procedure:\n{idx}\n\n"
        "List EVERY distinct step in the order it occurs. For each step give a short label "
        "and its start and end time in seconds, using the frame timestamps above.\n"
        "Respond with ONLY a JSON array, no prose:\n"
        '[{"step": "<short label>", "start": <seconds>, "end": <seconds>}, ...]'
    )


def ground(video, every=6, model_id=MODEL, max_tokens=1024):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    total = duration(video)
    with tempfile.TemporaryDirectory() as tmp:
        stamps = sample_frames(video, every, tmp)
        imgs = [f for _, f in stamps]
        model, processor = load(model_id)
        formatted = apply_chat_template(processor, model.config, build_prompt(stamps, total),
                                        num_images=len(imgs))
        out = generate(model, processor, formatted, image=imgs, max_tokens=max_tokens, verbose=False)
    out = getattr(out, "text", out)  # mlx-vlm >=0.6 returns a GenerationResult; older returns str
    s, e = out.find("["), out.rfind("]")  # salvage the array even if the model adds stray text
    return json.loads(out[s:e + 1]) if s >= 0 and e > s else []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ground.py path/to/procedure.mp4 [every_seconds]")
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(json.dumps(ground(sys.argv[1], every=every), indent=2))
