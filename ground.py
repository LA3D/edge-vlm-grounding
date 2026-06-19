"""Molmo2 (MLX) → ordered, timestamped step timeline for a procedure video.

Usage:
    python ground.py path/to/procedure.mp4

Output: JSON array of {"step", "start", "end"} in temporal order.
"""
import json, sys
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/Molmo2-8B-4bit"

PROMPT = (
    "You are watching a video of a laboratory / experimental procedure.\n"
    "List EVERY distinct step in the order it occurs. For each step give a short\n"
    "label and its start and end time in seconds.\n"
    "Respond with ONLY a JSON array, no prose:\n"
    '[{"step": "<short label>", "start": <seconds>, "end": <seconds>}, ...]'
)


def ground(video, model_id=MODEL, max_tokens=1024):
    model, processor = load(model_id)
    formatted = apply_chat_template(processor, model.config, PROMPT, num_videos=1)
    out = generate(model, processor, formatted, video=[video], max_tokens=max_tokens, verbose=False)
    s, e = out.find("["), out.rfind("]")  # salvage the array even if the model adds stray text
    return json.loads(out[s:e + 1]) if s >= 0 and e > s else []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ground.py path/to/procedure.mp4")
    print(json.dumps(ground(sys.argv[1]), indent=2))
