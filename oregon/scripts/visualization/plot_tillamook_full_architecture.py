#!/usr/bin/env python3
"""
Generate full architecture visualizations for the Tillamook Deep 7-channel U-Net.

Outputs:
- deep_7ch_architecture_full.svg  <- best for zooming / professor inspection
- deep_7ch_architecture_full.pdf
- deep_7ch_architecture_full.png
- deep_7ch_architecture_summary.txt

Requires:
    pip install torchview graphviz torchinfo

Also requires the Graphviz system executable (`dot`) to be installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from torchview import draw_graph
except ImportError as e:
    raise SystemExit(
        "Missing torchview. Install with:\n"
        "  python -m pip install torchview graphviz torchinfo"
    ) from e

try:
    from torchinfo import summary
except ImportError as e:
    raise SystemExit(
        "Missing torchinfo. Install with:\n"
        "  python -m pip install torchview graphviz torchinfo"
    ) from e

try:
    from train_tillamook_binary_feature_depth import DeepUNet
except ImportError as e:
    raise SystemExit(
        "Could not import DeepUNet from train_tillamook_binary_feature_depth.py.\n"
        "Put this script in the same F:\\LIDAR\\oregon directory as the trainer."
    ) from e


CHANNELS = [
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("model_architecture"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = DeepUNet(in_channels=len(CHANNELS)).cpu().eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model: DeepUNet")
    print("Input channels:", len(CHANNELS))
    print("Channels:", CHANNELS)
    print(f"Parameters: {total_params:,}")
    print(f"Trainable:  {trainable_params:,}")
    print(f"Input: ({args.batch_size}, {len(CHANNELS)}, {args.height}, {args.width})")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    summary_obj = summary(
        model,
        input_size=(args.batch_size, len(CHANNELS), args.height, args.width),
        depth=8,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ),
        row_settings=("var_names",),
        verbose=0,
        device="cpu",
    )

    summary_path = outdir / "deep_7ch_architecture_summary.txt"
    summary_path.write_text(
        "Tillamook Deep 7-channel U-Net\n"
        "================================\n\n"
        f"Input channels:\n  " + "\n  ".join(CHANNELS) + "\n\n"
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}\n\n"
        + str(summary_obj),
        encoding="utf-8",
    )

    print("Creating full graph. This can take a little while...")

    # ------------------------------------------------------------------
    # FULL graph from actual PyTorch computation.
    # expand_nested=True reveals Conv/BN/ReLU layers inside conv blocks.
    # ------------------------------------------------------------------
    graph = draw_graph(
        model,
        input_size=(args.batch_size, len(CHANNELS), args.height, args.width),
        device="cpu",
        expand_nested=True,
        depth=20,
        graph_name="Tillamook_DeepUNet_7ch",
        roll=False,
        save_graph=False,
    )

    dot = graph.visual_graph

    # Large portrait layout for zoomable SVG/PDF.
    dot.graph_attr.update({
        "rankdir": "TB",
        "ranksep": "0.45",
        "nodesep": "0.20",
        "splines": "ortho",
        "concentrate": "false",
        "bgcolor": "white",
        "pad": "0.25",
        "margin": "0.05",
        "dpi": "180",
    })

    # Avoid exotic fonts that previously caused Graphviz/Pango warnings.
    dot.node_attr.update({
        "fontname": "Arial",
        "fontsize": "9",
        "shape": "box",
        "margin": "0.06,0.04",
    })
    dot.edge_attr.update({
        "fontname": "Arial",
        "fontsize": "8",
        "arrowsize": "0.6",
    })

    base = outdir / "deep_7ch_architecture_full"

    # render() appends the extension itself.
    generated = []
    for fmt in ("svg", "pdf", "png"):
        try:
            rendered = dot.render(
                filename=base.name,
                directory=str(outdir),
                format=fmt,
                cleanup=True,
            )
            generated.append(rendered)
            print("Generated:", rendered)
        except Exception as e:
            print(f"WARNING: could not render {fmt}: {e}")

    # Save DOT source too, useful for later manual tweaking.
    dot_path = outdir / "deep_7ch_architecture_full.dot"
    dot_path.write_text(dot.source, encoding="utf-8")

    print()
    print("DONE")
    print("Best file for professor: deep_7ch_architecture_full.svg")
    print("Best printable file:     deep_7ch_architecture_full.pdf")
    print("Layer-by-layer summary:  deep_7ch_architecture_summary.txt")
    print("Graphviz source:         deep_7ch_architecture_full.dot")
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()
