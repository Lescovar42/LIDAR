from pathlib import Path

import torch
from torchinfo import summary
from torchview import draw_graph

from train_baseline import MiniUNet


CHECKPOINT = Path(
    "training_output_tillamook_15m_boundary_auto_control/best_model.pt"
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

channels = checkpoint["channels"]

model = MiniUNet(in_channels=len(channels))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

summary(
    model,
    input_size=(1, len(channels), 256, 256),
    col_names=("input_size", "output_size", "num_params"),
    depth=4,
)

graph = draw_graph(
    model,
    input_size=(1, len(channels), 256, 256),
    device="cpu",

    # IMPORTANT:
    expand_nested=True,
    depth=2,

    graph_name="Tillamook_MiniUNet",
)

g = graph.visual_graph

# Vertical architecture instead of ultra-wide strip
g.attr(
    rankdir="TB",
    dpi="200",
    nodesep="0.35",
    ranksep="0.55",
    pad="0.25",
)

g.attr(
    "node",
    fontname="Arial",
    fontsize="11",
)

g.attr(
    "edge",
    fontname="Arial",
    fontsize="9",
)

# Vector version — best for viewing/printing
g.render(
    filename="tillamook_minunet",
    format="svg",
    cleanup=True,
)

# Normal high-res image
g.render(
    filename="tillamook_minunet",
    format="png",
    cleanup=True,
)

print("Saved:")
print("  tillamook_minunet.svg")
print("  tillamook_minunet.png")