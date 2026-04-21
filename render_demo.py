from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from rendering import DeterministicCamouflageRenderer
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a camouflage image from a fused RGB background.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cm-per-pixel", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    render_cfg = dict(cfg.get("rendering", {}))
    if args.cm_per_pixel is not None:
        render_cfg["cm_per_pixel"] = args.cm_per_pixel

    renderer = DeterministicCamouflageRenderer(render_cfg)
    image = Image.open(args.input).convert("RGB")
    result = renderer.render(image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image).save(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
