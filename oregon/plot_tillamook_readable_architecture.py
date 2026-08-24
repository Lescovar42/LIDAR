#!/usr/bin/env python3
"""
Professor-readable architecture diagram for the Tillamook Deep 7-channel U-Net.

No Graphviz required.

Outputs:
    model_architecture_readable/
        deep_7ch_architecture_readable.svg
        deep_7ch_architecture_readable.pdf
        deep_7ch_architecture_readable.png
        deep_7ch_architecture_readable.txt

The diagram reflects the actual DeepUNet used in:
    train_tillamook_binary_feature_depth.py

Architecture:
    Input: 7 x 256 x 256

    Encoder:
      ConvBlock 7->32      256x256
      Pool
      ConvBlock 32->64     128x128
      Pool
      ConvBlock 64->128     64x64
      Pool
      ConvBlock 128->256    32x32
      Pool

    Bottleneck:
      ConvBlock 256->512    16x16

    Decoder:
      UpConv 512->256 + skip Enc4 -> ConvBlock 512->256
      UpConv 256->128 + skip Enc3 -> ConvBlock 256->128
      UpConv 128->64  + skip Enc2 -> ConvBlock 128->64
      UpConv 64->32   + skip Enc1 -> ConvBlock 64->32

    Head:
      1x1 Conv 32->1
      Landslide logits 1 x 256 x 256

Each ConvBlock:
    Conv3x3 -> BatchNorm -> ReLU
    Conv3x3 -> BatchNorm -> ReLU
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


CHANNELS = [
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
]


COLORS = {
    "ink": "#20303c",
    "muted": "#61717b",
    "paper": "#f5f7f4",
    "white": "#ffffff",
    "input": "#e1edf1",
    "encoder": "#dce9f4",
    "bottleneck": "#f5dfc6",
    "decoder": "#dcecdf",
    "output": "#f2d9dc",
    "skip": "#b46f5b",
}


def parameter_count():
    """Get parameter count from the actual project model if available."""
    try:
        from train_tillamook_binary_feature_depth import DeepUNet
        model = DeepUNet(in_channels=7)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        # Known count from the controlled 2x2 experiment.
        return 7_767_137


def box(ax, x, y, w, h, title, subtitle="", linewidth=1.6, role="encoder"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=linewidth,
        facecolor=COLORS[role],
        edgecolor=COLORS["ink"],
        zorder=3,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )

    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.2,
            color=COLORS["muted"],
            zorder=4,
        )

    return patch


def arrow(ax, start, end, connectionstyle="arc3", linewidth=1.5, linestyle="-", color=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color or COLORS["ink"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("model_architecture_readable"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    params = parameter_count()

    # Large landscape canvas; suitable for PDF/SVG and PowerPoint.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "DejaVu Sans"],
    })
    fig, ax = plt.subplots(figsize=(18, 10), facecolor=COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.suptitle(
        "Deep U-Net for LiDAR-Derived Landslide Segmentation",
        fontsize=19,
        fontweight="bold",
        y=0.97,
        color=COLORS["ink"],
    )

    ax.text(
        9,
        9.82,
        f"7 terrain channels • 256×256 patches • {params:,} trainable parameters",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------
    w = 2.25
    h = 1.05

    # Input
    input_x, input_y = 0.4, 7.4
    box(
        ax,
        input_x,
        input_y,
        w,
        h,
        "INPUT",
        "7 × 256 × 256",
        linewidth=2.0,
        role="input",
    )

    # Encoder
    enc = [
        (3.1, 7.4, "Encoder 1", "32 × 256 × 256"),
        (5.8, 5.9, "Encoder 2", "64 × 128 × 128"),
        (8.5, 4.4, "Encoder 3", "128 × 64 × 64"),
        (11.2, 2.9, "Encoder 4", "256 × 32 × 32"),
    ]

    for x, y, title, subtitle in enc:
        box(ax, x, y, w, h, title, subtitle, role="encoder")

    # Bottleneck
    bott_x, bott_y = 13.9, 1.4
    box(
        ax,
        bott_x,
        bott_y,
        w,
        h,
        "Bottleneck",
        "512 × 16 × 16",
        linewidth=2.0,
        role="bottleneck",
    )

    # Decoder climbs back to the left.
    dec = [
        (13.9, 4.1, "Decoder 4", "256 × 32 × 32"),
        (11.2, 5.6, "Decoder 3", "128 × 64 × 64"),
        (8.5, 7.1, "Decoder 2", "64 × 128 × 128"),
        (5.8, 8.6, "Decoder 1", "32 × 256 × 256"),
    ]

    for x, y, title, subtitle in dec:
        box(ax, x, y, w, h, title, subtitle, role="decoder")

    # Output
    output_x, output_y = 3.1, 8.6
    box(
        ax,
        output_x,
        output_y,
        w,
        h,
        "1×1 Conv → Output",
        "1 × 256 × 256 logits",
        linewidth=2.0,
        role="output",
    )

    # --------------------------------------------------------------
    # Main forward arrows
    # --------------------------------------------------------------
    arrow(
        ax,
        (input_x + w, input_y + h / 2),
        (enc[0][0], enc[0][1] + h / 2),
    )

    # Encoder downward staircase
    for i in range(3):
        x1, y1 = enc[i][0], enc[i][1]
        x2, y2 = enc[i + 1][0], enc[i + 1][1]

        arrow(
            ax,
            (x1 + w, y1 + h * 0.35),
            (x2, y2 + h * 0.65),
        )

        ax.text(
            (x1 + w + x2) / 2,
            (y1 + y2) / 2 + 0.1,
            "MaxPool 2×2",
            fontsize=7.8,
            ha="center",
            color=COLORS["muted"],
            bbox=dict(facecolor=COLORS["paper"], edgecolor="none", pad=1.5),
        )

    # Encoder 4 -> bottleneck
    arrow(
        ax,
        (enc[3][0] + w, enc[3][1] + h * 0.35),
        (bott_x, bott_y + h * 0.65),
    )
    ax.text(
        13.55,
        2.35,
        "MaxPool 2×2",
        fontsize=7.8,
        ha="center",
        color=COLORS["muted"],
        bbox=dict(facecolor=COLORS["paper"], edgecolor="none", pad=1.5),
    )

    # Bottleneck -> Decoder 4
    arrow(
        ax,
        (bott_x + w / 2, bott_y + h),
        (dec[0][0] + w / 2, dec[0][1]),
    )
    ax.text(
        bott_x + w / 2 + 0.25,
        3.25,
        "UpConv 2×2",
        fontsize=7.8,
        rotation=90,
        va="center",
        color=COLORS["muted"],
        bbox=dict(facecolor=COLORS["paper"], edgecolor="none", pad=1.5),
    )

    # Decoder upward staircase
    for i in range(3):
        x1, y1 = dec[i][0], dec[i][1]
        x2, y2 = dec[i + 1][0], dec[i + 1][1]

        arrow(
            ax,
            (x1, y1 + h * 0.65),
            (x2 + w, y2 + h * 0.35),
        )

        ax.text(
            (x1 + x2 + w) / 2,
            (y1 + y2) / 2 + 0.25,
            "UpConv 2×2",
            fontsize=7.8,
            ha="center",
            color=COLORS["muted"],
            bbox=dict(facecolor=COLORS["paper"], edgecolor="none", pad=1.5),
        )

    # Decoder 1 -> output
    arrow(
        ax,
        (dec[3][0], dec[3][1] + h / 2),
        (output_x + w, output_y + h / 2),
    )

    # --------------------------------------------------------------
    # Skip connections
    # Dashed, routed around the central diagram.
    # --------------------------------------------------------------
    skip_pairs = [
        # Encoder1 -> Decoder1
        (
            (enc[0][0] + w / 2, enc[0][1] + h),
            (dec[3][0] + w / 2, dec[3][1]),
            "Skip 1",
            -0.12,
        ),
        # Encoder2 -> Decoder2
        (
            (enc[1][0] + w / 2, enc[1][1] + h),
            (dec[2][0] + w / 2, dec[2][1]),
            "Skip 2",
            -0.12,
        ),
        # Encoder3 -> Decoder3
        (
            (enc[2][0] + w / 2, enc[2][1] + h),
            (dec[1][0] + w / 2, dec[1][1]),
            "Skip 3",
            -0.12,
        ),
        # Encoder4 -> Decoder4
        (
            (enc[3][0] + w / 2, enc[3][1] + h),
            (dec[0][0] + w / 2, dec[0][1]),
            "Skip 4",
            -0.12,
        ),
    ]

    for start, end, label, rad in skip_pairs:
        arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=1.2,
            linestyle="--",
            color=COLORS["skip"],
        )

        ax.text(
            (start[0] + end[0]) / 2 + 0.15,
            (start[1] + end[1]) / 2,
            label + " / concatenate",
            fontsize=7.2,
            rotation=90,
            ha="center",
            va="center",
            color=COLORS["skip"],
            bbox=dict(facecolor=COLORS["paper"], edgecolor="none", pad=1.5),
        )

    # --------------------------------------------------------------
    # ConvBlock legend / explanation
    # --------------------------------------------------------------
    legend_x, legend_y = 0.5, 0.55
    legend_w, legend_h = 6.9, 2.0

    legend = FancyBboxPatch(
        (legend_x, legend_y),
        legend_w,
        legend_h,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        facecolor=COLORS["white"],
        edgecolor="#c8d2d0",
        zorder=2,
    )
    ax.add_patch(legend)

    ax.text(
        legend_x + 0.25,
        legend_y + legend_h - 0.35,
        "Each Encoder / Decoder ConvBlock",
        fontsize=10,
        fontweight="bold",
        va="center",
        color=COLORS["ink"],
    )

    ax.text(
        legend_x + 0.25,
        legend_y + 1.02,
        "Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm → ReLU",
        fontsize=9,
        va="center",
        color=COLORS["ink"],
    )

    ax.text(
        legend_x + 0.25,
        legend_y + 0.45,
        "Decoder input = upsampled decoder features concatenated with same-resolution encoder features.",
        fontsize=8.2,
        va="center",
        color=COLORS["muted"],
    )

    # Input channel list
    channels_text = (
        "Input channels:\n"
        "local_relief • slope_degrees • aspect_sin • aspect_cos\n"
        "curvature • multidirectional_hillshade • TRI"
    )

    ax.text(
        8.2,
        0.95,
        channels_text,
        fontsize=8.5,
        va="bottom",
        color=COLORS["muted"],
    )

    # Footer
    ax.text(
        17.5,
        0.25,
        "Tillamook strict-binary development model",
        fontsize=7.5,
        ha="right",
        color=COLORS["muted"],
    )

    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.95))

    base = outdir / "deep_7ch_architecture_readable"

    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")

    plt.close(fig)

    summary = f"""Deep U-Net — Tillamook LiDAR landslide segmentation
====================================================

Input:
    7 x 256 x 256

Terrain channels:
    1. local_relief
    2. slope_degrees
    3. aspect_sin
    4. aspect_cos
    5. curvature
    6. multidirectional_hillshade
    7. tri

Encoder:
    Encoder 1: 7 -> 32 channels, 256 x 256
    MaxPool
    Encoder 2: 32 -> 64 channels, 128 x 128
    MaxPool
    Encoder 3: 64 -> 128 channels, 64 x 64
    MaxPool
    Encoder 4: 128 -> 256 channels, 32 x 32
    MaxPool

Bottleneck:
    256 -> 512 channels, 16 x 16

Decoder:
    UpConv 512 -> 256
    concatenate Encoder 4
    ConvBlock 512 -> 256

    UpConv 256 -> 128
    concatenate Encoder 3
    ConvBlock 256 -> 128

    UpConv 128 -> 64
    concatenate Encoder 2
    ConvBlock 128 -> 64

    UpConv 64 -> 32
    concatenate Encoder 1
    ConvBlock 64 -> 32

Head:
    1x1 Conv: 32 -> 1
    output: 1 x 256 x 256 landslide logits

Each ConvBlock:
    Conv3x3 -> BatchNorm -> ReLU
    Conv3x3 -> BatchNorm -> ReLU

Parameters:
    {params:,}

Segmentation:
    sigmoid(logit) -> probability
    validation-selected threshold -> binary landslide prediction
"""

    base.with_suffix(".txt").write_text(summary, encoding="utf-8")

    print("DONE")
    print("Professor-readable architecture generated:")
    print("  ", base.with_suffix(".svg"))
    print("  ", base.with_suffix(".pdf"))
    print("  ", base.with_suffix(".png"))
    print("  ", base.with_suffix(".txt"))
    print()
    print(f"Parameters: {params:,}")
    print("No Graphviz / dot required.")


if __name__ == "__main__":
    main()
