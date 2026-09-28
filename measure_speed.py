"""Measure forward-pass latency of a saved reference forecaster."""
import argparse
import json
import time
from pathlib import Path

import torch

from forecast import positive_int
from models.transformer import TransformerForecaster


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=positive_int, default=2)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if saved.get("format") != "bird-reference-v1":
        raise ValueError("Expected a checkpoint produced by forecast.py.")
    model = TransformerForecaster(**saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["model"], strict=True)
    sample = torch.zeros(1, saved["model_config"]["past_len"], 2, device=args.device)

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            model(sample)
            synchronize()
        seconds = time.perf_counter() - started
    print(json.dumps(dict(
        scope="Reference model forward pass on synthetic input; excludes detection, tracking and I/O.",
        device=args.device, torch_version=torch.__version__, batch_size=1,
        iterations=args.iterations, warmup=args.warmup,
        past_len=model.P, future_len=model.M,
        parameters=sum(p.numel() for p in model.parameters()),
        milliseconds_per_call=1000 * seconds / args.iterations,
        calls_per_second=args.iterations / seconds,
    ), indent=2))


if __name__ == "__main__":
    main()
