# -*- coding: utf-8 -*-
"""Presentation-quality visualization helpers for FVNB-MNR samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_sample_grid(sample_path: str | Path, output_path: str | Path) -> Path:
    sample_path = Path(sample_path)
    output_path = Path(output_path)
    loaded = np.load(sample_path, allow_pickle=False)
    context_images = loaded["context_images"]
    candidate_images = loaded["answer_set_images"]
    correct_index = int(loaded["correct_answer_image_index"])
    metadata = json.loads(str(loaded["metadata_json"]))

    panel_size = int(context_images.shape[-1])
    scale = 2 if panel_size >= 120 else 3
    tile = panel_size * scale
    gap = 28
    label_h = 58
    header_h = 94
    cols = 4
    rows = 3
    width = cols * tile + (cols + 1) * gap
    height = header_h + rows * (tile + label_h) + rows * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    label_font = _load_font(13)
    small_font = _load_font(11)

    rule = metadata["rule"]
    validity = metadata["validity"]
    draw.text((gap, 18), "FVNB-MNR symbolic sample", fill=(20, 20, 20), font=title_font)
    draw.text(
        (gap, 46),
        "rule={0}  label={1}  margin={2:.3f}  max_neg={3:.3f}".format(
            rule["id"],
            correct_index,
            float(validity["top_margin"]),
            float(validity["max_negative_score"]),
        ),
        fill=(46, 46, 46),
        font=label_font,
    )
    draw.text((gap, 66), "C=fuzzy truth, H=hard-role truth", fill=(86, 86, 86), font=small_font)

    for index, arr in enumerate(context_images):
        x = gap + index * (tile + gap)
        y = header_h
        panel_meta = metadata["context"][index]
        _paste_tile(
            canvas,
            draw,
            arr,
            x,
            y,
            tile,
            "Context {0}".format(index),
            "C={0:.3f}  H={1:.1f}".format(panel_meta["truth_score"], panel_meta["hard_truth_score"]),
            label_font,
            small_font,
            border=(124, 124, 124),
        )

    for index, arr in enumerate(candidate_images):
        row = index // cols
        col = index % cols
        x = gap + col * (tile + gap)
        y = header_h + (tile + label_h + gap) + row * (tile + label_h + gap)
        candidate = metadata["candidates"][index]
        negative_type = candidate["negative_type"] or "correct"
        is_correct = index == correct_index
        border = (24, 132, 60) if is_correct else (216, 78, 55)
        if negative_type == "hard_role_invariant_fuzzy_flip":
            border = (222, 145, 28)
        title = "Candidate {0}{1}".format(index, "  ✓" if is_correct else "")
        subtitle = "{0} | C={1:.3f} H={2:.1f}".format(
            _short_type(negative_type),
            float(candidate["truth_score"]),
            float(candidate["hard_truth_score"]),
        )
        _paste_tile(canvas, draw, arr, x, y, tile, title, subtitle, label_font, small_font, border=border)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _paste_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    arr: np.ndarray,
    x: int,
    y: int,
    tile: int,
    title: str,
    subtitle: str,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    border: tuple[int, int, int],
) -> None:
    draw.text((x, y), title, fill=(18, 18, 18), font=label_font)
    draw.text((x, y + 20), subtitle, fill=(70, 70, 70), font=small_font)
    panel = Image.fromarray(arr).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
    panel_y = y + 42
    canvas.paste(panel, (x, panel_y))
    for offset in range(3):
        draw.rectangle([x - offset, panel_y - offset, x + tile + offset, panel_y + tile + offset], outline=border)


def _short_type(value: str) -> str:
    replacements = {
        "same_number_fuzzy_role_shift": "same-number shift",
        "same_coordinate_fuzzy_boundary_shift": "same-coordinate shift",
        "hard_role_invariant_fuzzy_flip": "hard-role fuzzy flip",
        "fuzzy_rule_false_positive": "fuzzy false positive",
        "shortcut_consistent_false_positive": "shortcut false positive",
        "arithmetic_only_distractor": "arithmetic distractor",
        "near_miss_distractor": "near miss",
    }
    return replacements.get(value, value)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an FVNB-MNR .npz sample as an overview PNG.")
    parser.add_argument("sample", type=str, help="Path to a generated FVNB-MNR .npz sample.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults to <sample>_grid.png.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sample_path = Path(args.sample)
    output = Path(args.output) if args.output else sample_path.with_name(sample_path.stem + "_grid.png")
    saved = save_sample_grid(sample_path, output)
    print(saved)


if __name__ == "__main__":
    main()
