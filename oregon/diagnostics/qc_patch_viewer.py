#!/usr/bin/env python3
"""Interactive one-patch-at-a-time QC viewer for Oregon LiDAR/SLIDO.

Reads a dataset created by ``build_dataset.py`` and writes QC decisions directly
into ``patches_qc.csv``. Every decision is saved immediately and atomically.

Keyboard shortcuts
------------------
A  accept
B  accept_approximate_boundary
M  reject_misaligned
V  reject_not_visible
E  reject_engineered_landform
D  reject_bad_dem
U  clear / unreviewed

Right / Space / PageDown   next patch
Left / Backspace / PageUp  previous patch
Ctrl+S                     save
Ctrl+F                     search patch ID
Q                          save and quit

Example
-------
python diagnostics/qc_patch_viewer.py --dataset-dir dataset_pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:
    raise SystemExit(
        "Tkinter is unavailable. On Windows, reinstall Python from python.org "
        "with 'tcl/tk and IDLE' enabled."
    ) from exc

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


QC_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "accept", "Accept"),
    ("B", "accept_approximate_boundary", "Accept approximate boundary"),
    ("M", "reject_misaligned", "Reject: misaligned"),
    ("V", "reject_not_visible", "Reject: not visible / unclear"),
    ("E", "reject_engineered_landform", "Reject: engineered landform"),
    ("D", "reject_bad_dem", "Reject: bad DEM"),
    ("U", "", "Clear / unreviewed"),
)
ACCEPTED_QC = {"accept", "accept_approximate_boundary"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ensure_qc_fields(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    fields = list(rows[0])
    for field in ("qc_status", "qc_notes"):
        if field not in fields:
            fields.append(field)
        for row in rows:
            row.setdefault(field, "")
    return fields


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write beside the destination, fsync, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def import_review(rows: list[dict[str, str]], review_path: Path) -> int:
    """Import non-empty decisions from the existing sampled QC CSV."""
    if not review_path.exists():
        return 0
    decisions = {
        row.get("patch_id", ""): (
            row.get("qc_status", "").strip(),
            row.get("qc_notes", "").strip(),
        )
        for row in read_csv(review_path)
        if row.get("patch_id") and row.get("qc_status", "").strip()
    }
    count = 0
    for row in rows:
        decision = decisions.get(row.get("patch_id", ""))
        if decision and not row.get("qc_status", "").strip():
            row["qc_status"], row["qc_notes"] = decision
            count += 1
    return count


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    values = np.asarray(array, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def parse_bool(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes", "y"}


def select_indices(
    rows: list[dict[str, str]],
    *,
    split: str,
    category: str,
    only_unreviewed: bool,
    shuffle: bool,
    seed: int,
) -> list[int]:
    selected = []
    for index, row in enumerate(rows):
        if split != "all" and row.get("split") != split:
            continue
        if category != "all" and row.get("category") != category:
            continue
        if only_unreviewed and row.get("qc_status", "").strip():
            continue
        selected.append(index)
    if shuffle:
        random.Random(seed).shuffle(selected)
    return selected


def first_unreviewed(rows: list[dict[str, str]], indices: list[int]) -> int:
    for position, index in enumerate(indices):
        if not rows[index].get("qc_status", "").strip():
            return position
    return 0


class PatchQCViewer:
    def __init__(
        self,
        *,
        root: tk.Tk,
        dataset_dir: Path,
        output_path: Path,
        rows: list[dict[str, str]],
        fields: list[str],
        channels: list[str],
        indices: list[int],
        start_position: int,
    ) -> None:
        self.root = root
        self.dataset_dir = dataset_dir
        self.output_path = output_path
        self.rows = rows
        self.fields = fields
        self.channels = channels
        self.channel_index = {name: index for index, name in enumerate(channels)}
        self.indices = indices
        self.position = start_position
        self.current_index: int | None = None
        self.notes_dirty = False

        required = {"slope_degrees", "multidirectional_hillshade"}
        missing = sorted(required - set(self.channel_index))
        if missing:
            raise ValueError(f"Missing channels {missing}; found {channels}")

        self.progress_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.metadata_var = tk.StringVar()
        self.search_var = tk.StringVar()

        self.root.title("Oregon LiDAR / SLIDO Patch QC")
        self.root.geometry("1500x930")
        self.root.minsize(1050, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        self.bind_keys()
        self.show_current()

    @property
    def row(self) -> dict[str, str]:
        return self.rows[self.indices[self.position]]

    def build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.progress_var, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(20, 0))

        search = ttk.Frame(top)
        search.pack(side=tk.RIGHT)
        ttk.Label(search, text="Patch ID:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search, textvariable=self.search_var, width=42)
        self.search_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(search, text="Go", command=self.go_to_patch).pack(side=tk.LEFT)

        ttk.Label(
            self.root,
            textvariable=self.metadata_var,
            justify=tk.LEFT,
            anchor=tk.W,
            padding=(10, 0, 10, 5),
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        self.figure = Figure(figsize=(14, 7), dpi=100, constrained_layout=True)
        self.axes = [self.figure.add_subplot(1, 4, i + 1) for i in range(4)]
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)

        notes_frame = ttk.LabelFrame(self.root, text="QC notes", padding=6)
        notes_frame.pack(fill=tk.X, padx=8, pady=4)
        self.notes = tk.Text(notes_frame, height=3, wrap=tk.WORD, undo=True)
        self.notes.pack(fill=tk.X)
        self.notes.bind("<<Modified>>", self.notes_modified)

        buttons = ttk.Frame(self.root, padding=(8, 2, 8, 8))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="◀ Previous", command=self.previous).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Next ▶", command=self.next).pack(side=tk.LEFT, padx=(5, 15))
        for key, status, label in QC_OPTIONS:
            ttk.Button(
                buttons,
                text=f"[{key}] {label}",
                command=lambda value=status: self.decide(value),
            ).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons, text="Save", command=self.save).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(buttons, text="Quit", command=self.close).pack(side=tk.RIGHT)

    def bind_keys(self) -> None:
        for key, status, _ in QC_OPTIONS:
            self.root.bind(
                f"<KeyPress-{key.casefold()}>",
                lambda event, value=status: self.decision_shortcut(event, value),
            )
        self.root.bind("<Right>", lambda event: self.next())
        self.root.bind("<space>", lambda event: self.next())
        self.root.bind("<Next>", lambda event: self.next())
        self.root.bind("<Left>", lambda event: self.previous())
        self.root.bind("<BackSpace>", lambda event: self.previous())
        self.root.bind("<Prior>", lambda event: self.previous())
        self.root.bind("<Control-s>", lambda event: self.save())
        self.root.bind("<Control-f>", self.focus_search)
        self.root.bind("<KeyPress-q>", self.quit_shortcut)
        self.root.bind("<Return>", self.enter_key)

    def text_focused(self) -> bool:
        return self.root.focus_get() in {self.notes, self.search_entry}

    def decision_shortcut(self, event: tk.Event, status: str) -> str | None:
        if self.text_focused():
            return None
        self.decide(status)
        return "break"

    def quit_shortcut(self, event: tk.Event) -> str | None:
        if self.text_focused():
            return None
        self.close()
        return "break"

    def enter_key(self, event: tk.Event) -> str | None:
        if self.root.focus_get() == self.search_entry:
            self.go_to_patch()
            return "break"
        return None

    def focus_search(self, event: tk.Event | None = None) -> str:
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, tk.END)
        return "break"

    def notes_modified(self, event: tk.Event | None = None) -> None:
        if self.notes.edit_modified():
            self.notes_dirty = True
            self.notes.edit_modified(False)

    def capture_notes(self) -> None:
        if self.current_index is None:
            return
        value = self.notes.get("1.0", tk.END).strip()
        if self.rows[self.current_index].get("qc_notes", "") != value:
            self.rows[self.current_index]["qc_notes"] = value
            self.notes_dirty = True

    def save(self) -> None:
        self.capture_notes()
        atomic_write_csv(self.output_path, self.rows, self.fields)
        self.notes_dirty = False
        self.update_labels(saved=True)

    def update_labels(self, saved: bool = False) -> None:
        statuses = [row.get("qc_status", "").strip().casefold() for row in self.rows]
        reviewed = sum(bool(value) for value in statuses)
        accepted = sum(value in ACCEPTED_QC for value in statuses)
        rejected = sum(value.startswith("reject_") for value in statuses)
        saved_text = " | saved" if saved else ""
        self.progress_var.set(
            f"Patch {self.position + 1}/{len(self.indices)} | reviewed {reviewed}/{len(self.rows)} "
            f"| accepted {accepted} | rejected {rejected}{saved_text}"
        )
        self.status_var.set(f"Current QC: {self.row.get('qc_status', '').strip() or 'unreviewed'}")

    def decide(self, status: str) -> None:
        self.capture_notes()
        self.row["qc_status"] = status
        atomic_write_csv(self.output_path, self.rows, self.fields)
        self.notes_dirty = False
        if self.position < len(self.indices) - 1:
            self.position += 1
            self.show_current()
        else:
            self.update_labels(saved=True)
            messagebox.showinfo("QC complete", f"Reached the final selected patch.\n\nSaved to:\n{self.output_path}")

    def previous(self) -> None:
        self.capture_notes()
        if self.notes_dirty:
            self.save()
        if self.position > 0:
            self.position -= 1
            self.show_current()

    def next(self) -> None:
        self.capture_notes()
        if self.notes_dirty:
            self.save()
        if self.position < len(self.indices) - 1:
            self.position += 1
            self.show_current()

    def go_to_patch(self) -> None:
        query = self.search_var.get().strip().casefold()
        if not query:
            return
        self.capture_notes()
        for position, index in enumerate(self.indices):
            if query in self.rows[index].get("patch_id", "").casefold():
                self.position = position
                self.show_current()
                return
        messagebox.showwarning("Not found", f"No selected patch contains:\n{query}")

    def show_current(self) -> None:
        self.current_index = self.indices[self.position]
        row = self.row
        patch_path = self.dataset_dir / row["patch_path"]
        if not patch_path.exists():
            messagebox.showerror("Missing patch", f"Patch file does not exist:\n{patch_path}")
            return

        try:
            with np.load(patch_path) as data:
                features = data["features"].astype(np.float32)
                mask = data["mask"].astype(np.uint8)
        except Exception as exc:
            messagebox.showerror("Cannot load patch", f"{patch_path}\n\n{type(exc).__name__}: {exc}")
            return

        hillshade = features[self.channel_index["multidirectional_hillshade"]]
        slope = features[self.channel_index["slope_degrees"]]
        if "local_relief" in self.channel_index:
            relief = features[self.channel_index["local_relief"]]
            relief_title = "Local relief"
        else:
            relief = hillshade
            relief_title = "Hillshade duplicate"

        for axis in self.axes:
            axis.clear()
            axis.set_axis_off()

        hmin, hmax = robust_limits(hillshade)
        rmin, rmax = robust_limits(relief)
        slope_max = max(45.0, float(np.nanpercentile(slope, 99)))

        self.axes[0].imshow(hillshade, cmap="gray", vmin=hmin, vmax=hmax)
        self.axes[0].set_title("Multidirectional hillshade")
        self.axes[1].imshow(relief, cmap="gray", vmin=rmin, vmax=rmax)
        self.axes[1].set_title(relief_title)
        self.axes[2].imshow(slope, cmap="magma", vmin=0, vmax=slope_max)
        self.axes[2].set_title("Slope (degrees)")
        self.axes[3].imshow(hillshade, cmap="gray", vmin=hmin, vmax=hmax)
        overlay = np.ma.masked_where(mask == 0, mask)
        self.axes[3].imshow(overlay, cmap="Reds", alpha=0.50, vmin=0, vmax=1)
        if mask.min() != mask.max():
            self.axes[3].contour(mask, levels=[0.5], colors="red", linewidths=1.1)
        self.axes[3].set_title("SLIDO ground-truth overlay")

        self.figure.suptitle(f"{row.get('patch_id', '')}\n{row.get('tile_name', '')}", fontsize=12)
        self.canvas.draw_idle()

        self.notes.delete("1.0", tk.END)
        self.notes.insert("1.0", row.get("qc_notes", ""))
        self.notes.edit_modified(False)
        self.notes_dirty = False
        self.search_var.set(row.get("patch_id", ""))

        positive = float(row.get("positive_fraction") or mask.mean())
        ground = float(row.get("ground_fraction") or 0)
        mean_slope = float(row.get("mean_slope_degrees") or np.nanmean(slope))
        self.metadata_var.set(
            f"split={row.get('split', '')}    category={row.get('category', '')}    "
            f"positive={100 * positive:.2f}%    ground={100 * ground:.1f}%    "
            f"mean slope={mean_slope:.1f}°    hard negative={parse_bool(row.get('is_hard_negative', ''))}\n"
            f"row/col={row.get('row_offset', '')}/{row.get('col_offset', '')}    "
            f"CRS={row.get('crs', '')}    distance to positive={row.get('distance_to_positive_m', '')} m\n"
            f"SLIDO refs in tile={row.get('slido_ref_ids_in_tile', '')}"
        )
        self.update_labels()

    def close(self) -> None:
        try:
            self.save()
        except Exception as exc:
            leave = messagebox.askyesno(
                "Save failed",
                f"Could not save QC file:\n\n{type(exc).__name__}: {exc}\n\nQuit without saving?",
            )
            if not leave:
                return
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Oregon LiDAR/SLIDO patches one by one.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Input CSV. Default: patches_qc.csv if present, otherwise patches.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. Default: DATASET_DIR/patches_qc.csv.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Existing qc_review.csv to import. Default: DATASET_DIR/qc/qc_review.csv.",
    )
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument(
        "--category",
        default="all",
        help="Filter category: positive_interior, positive_boundary, negative, etc.",
    )
    parser.add_argument("--only-unreviewed", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    qc_manifest = dataset_dir / "patches_qc.csv"
    default_manifest = qc_manifest if qc_manifest.exists() else dataset_dir / "patches.csv"
    manifest_path = (args.manifest or default_manifest).resolve()
    output_path = (args.out or qc_manifest).resolve()
    channels_path = dataset_dir / "channels.json"

    if not dataset_dir.exists():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")
    if not manifest_path.exists():
        parser.error(f"Manifest does not exist: {manifest_path}")
    if not channels_path.exists():
        parser.error(f"Missing channel definition: {channels_path}")

    rows = read_csv(manifest_path)
    if not rows:
        parser.error(f"Manifest contains no rows: {manifest_path}")
    fields = ensure_qc_fields(rows)

    review_path = args.review.resolve() if args.review else dataset_dir / "qc" / "qc_review.csv"
    imported = import_review(rows, review_path)
    if imported:
        print(f"Imported {imported} decision(s) from {review_path}")

    channels = json.loads(channels_path.read_text(encoding="utf-8")).get("feature_names", [])
    if not channels:
        parser.error(f"No feature_names found in {channels_path}")

    indices = select_indices(
        rows,
        split=args.split,
        category=args.category,
        only_unreviewed=args.only_unreviewed,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not indices:
        parser.error("No patches match the selected filters")

    atomic_write_csv(output_path, rows, fields)
    start = first_unreviewed(rows, indices)
    counts = Counter(row.get("qc_status", "").strip() or "unreviewed" for row in rows)
    print(f"Loaded {len(rows)} patch(es); showing {len(indices)}")
    print(f"QC status counts: {dict(counts)}")
    print(f"Saving decisions to: {output_path}")
    print("Keyboard: A/B/M/V/E/D/U, arrows, Ctrl+S, Q")

    root = tk.Tk()
    try:
        PatchQCViewer(
            root=root,
            dataset_dir=dataset_dir,
            output_path=output_path,
            rows=rows,
            fields=fields,
            channels=channels,
            indices=indices,
            start_position=start,
        )
    except Exception as exc:
        root.destroy()
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc

    root.mainloop()
    print(f"QC progress saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
