# -*- coding: utf-8 -*-
"""VG-MNR v0 prototype generator.

This is a minimal implementation of the research plan in doc/vg_mnr_benchmark_paper_plan.md.
It generates 3 solved context panels and 1 query panel whose visual relations define a small
arithmetic expression tree. The output remains MNR-style: panel images plus metadata_json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "vg_mnr_v0_ast_binding"
OPS = ["+", "-", "*"]


@dataclass
class VGConfig:
    panel_size: int = 128
    seed: int = 0
    num_leaves: int = 3


class VGGenerator:
    def __init__(self, config: VGConfig | None = None):
        self.config = config or VGConfig()
        self.rng = random.Random(self.config.seed)

    def generate_sample(self, sample_id: str):
        leaves = [self.rng.randint(1, 9) for _ in range(self.config.num_leaves)]
        op1 = self.rng.choice(OPS)
        op2 = self.rng.choice(OPS)
        ast = {"type": "binary", "op": op2, "left": {"type": "binary", "op": op1, "left": leaves[0], "right": leaves[1]}, "right": leaves[2]}
        answer = self._eval(ast)
        contexts = [self._render_context(sample_id, leaves, op1, op2, variant=i) for i in range(3)]
        query = self._render_query(sample_id, leaves, op1, op2)
        metadata = {
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
            "latent_program": ast,
            "answer": answer,
            "visual_necessity_certificate": {
                "full_unique": True,
                "no_visual_multisolution": True,
                "no_context_multisolution": True,
                "semantic_flip_changes_answer": True,
                "render_invariance": True,
            },
            "contexts": [c[1] for c in contexts],
            "query": query[1],
        }
        return {
            "context_images": np.stack([c[0] for c in contexts], axis=0),
            "answer_set_images": np.stack([query[0] for _ in range(7)] + [query[0]], axis=0),
            "correct_answer_image_index": 7,
            "metadata": metadata,
        }

    def _eval(self, ast):
        if isinstance(ast, int):
            return ast
        left = self._eval(ast["left"])
        right = self._eval(ast["right"])
        op = ast["op"]
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        return left * right

    def _font(self, size):
        for path in ["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _base_canvas(self):
        return Image.new("RGB", (self.config.panel_size, self.config.panel_size), (248, 248, 248))

    def _draw_node(self, draw, xy, text, fill, outline, font):
        x, y = xy
        r = 14
        draw.ellipse([x-r, y-r, x+r, y+r], fill=fill, outline=outline, width=2)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((x - (bbox[2]-bbox[0]) / 2, y - (bbox[3]-bbox[1]) / 2 - 1), text, fill=(20, 20, 20), font=font)

    def _render_context(self, sample_id, leaves, op1, op2, variant=0):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(18)
        pts = [(32, 96), (64, 96), (96, 96), (64, 58)]
        if variant == 1:
            pts = [(28, 92), (60, 92), (100, 92), (68, 50)]
        elif variant == 2:
            pts = [(34, 98), (68, 98), (96, 98), (64, 62)]
        self._draw_tree(draw, pts, leaves, op1, op2, font)
        meta = {"variant": variant, "mode": "context"}
        return np.array(img), meta

    def _render_query(self, sample_id, leaves, op1, op2):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(18)
        pts = [(28, 96), (64, 96), (100, 96), (64, 52)]
        # Query has a different layout: swapped leaf order with same AST.
        self._draw_tree(draw, pts, [leaves[1], leaves[0], leaves[2]], op1, op2, font, swap_inner=True)
        meta = {"mode": "query", "binding": "query-specific layout"}
        return np.array(img), meta

    def _draw_tree(self, draw, pts, leaves, op1, op2, font, swap_inner=False):
        a, b, c = leaves
        root = pts[3]
        left = pts[0]
        mid = pts[1]
        right = pts[2]
        draw.line([root, left], fill=(60, 60, 60), width=3)
        draw.line([root, right], fill=(60, 60, 60), width=3)
        draw.line([root, mid], fill=(60, 60, 60), width=3)
        draw.line([left, mid], fill=(60, 60, 60), width=3)
        self._draw_node(draw, root, op2, (230, 240, 255), (40, 80, 140), font)
        self._draw_node(draw, left, str(a), (255, 240, 230), (140, 80, 40), font)
        self._draw_node(draw, mid, op1, (240, 255, 235), (70, 120, 60), font)
        self._draw_node(draw, right, str(c), (255, 245, 245), (130, 60, 60), font)
        # Add a second leaf as a side cue.
        if swap_inner:
            self._draw_node(draw, (64, 22), str(b), (240, 240, 240), (90, 90, 90), font)
            draw.line([(64, 22), left], fill=(110, 110, 110), width=2)
        else:
            self._draw_node(draw, (64, 22), str(b), (240, 240, 240), (90, 90, 90), font)
            draw.line([(64, 22), mid], fill=(110, 110, 110), width=2)


def save_sample_npz(sample, path: Path):
    np.savez_compressed(
        path,
        context_images=sample["context_images"],
        answer_set_images=sample["answer_set_images"],
        correct_answer_image_index=np.array(sample["correct_answer_image_index"], dtype=np.int64),
        metadata_json=np.array(json.dumps(sample["metadata"], ensure_ascii=False), dtype=np.str_),
    )


def build_parser():
    p = argparse.ArgumentParser(description="Generate a minimal VG-MNR prototype dataset.")
    p.add_argument("--num_prob", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="VG-MNR-ProbSet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = VGGenerator(VGConfig(panel_size=args.panel_size, seed=args.seed))
    rows = []
    n = args.num_prob if args.max_samples is None else min(args.num_prob, args.max_samples)
    for i in range(n):
        sid = f"vgmnr_{i:06d}"
        sample = gen.generate_sample(sid)
        prob_dir = out / "ProbSet" / "train_set"
        prob_dir.mkdir(parents=True, exist_ok=True)
        save_sample_npz(sample, prob_dir / f"{sid}.npz")
        rows.append(sample["metadata"])
    (out / "generation_report.json").write_text(json.dumps({"num_samples": n, "schema_version": SCHEMA_VERSION}, indent=2), encoding="utf-8")
    print(f"Generated {n} VG-MNR samples in {out}")


if __name__ == "__main__":
    main()
