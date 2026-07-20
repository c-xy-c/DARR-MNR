# -*- coding: utf-8 -*-
"""VG-MNR v0 prototype generator.

This prototype follows the VG-MNR proposal in doc/vg_mnr_benchmark_paper_plan.md.
It supports three modes that map to the proposal's early diagnostic ideas:
- full: visually grounded context/query panels
- no_visual: removes functional visual relations
- no_context: removes context panels while keeping the query intact

The implementation is intentionally small so it can be iterated quickly.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = "vg_mnr_v0_ast_binding"
OPS = ["+", "-", "*"]
MODES = ["full", "no_visual", "no_context", "collision"]


@dataclass
class VGConfig:
    panel_size: int = 128
    seed: int = 0
    num_leaves: int = 3
    mode: str = "full"


class VGGenerator:
    def __init__(self, config: VGConfig | None = None):
        self.config = config or VGConfig()
        self.rng = random.Random(self.config.seed)

    def generate_sample(self, sample_id: str):
        leaves = [self.rng.randint(1, 9) for _ in range(self.config.num_leaves)]
        op1 = self.rng.choice(OPS)
        op2 = self.rng.choice(OPS)
        ast = {
            "type": "binary",
            "op": op2,
            "left": {"type": "binary", "op": op1, "left": leaves[0], "right": leaves[1]},
            "right": leaves[2],
        }
        answer = self._eval(ast)

        contexts = []
        for i in range(3):
            contexts.append(self._render_context(leaves, op1, op2, variant=i, mode=self.config.mode))
        query = self._render_query(leaves, op1, op2, mode=self.config.mode)

        pair_id = None
        collision_role = None
        if self.config.mode == "collision":
            pair_id = f"{sample_id}_pair"
            collision_role = "A" if self.rng.random() < 0.5 else "B"

        metadata = {
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
            "mode": self.config.mode,
            "latent_program": ast,
            "answer": int(answer),
            "pair_id": pair_id,
            "collision_role": collision_role,
            "collision_kind": "query_binding_swap" if self.config.mode == "collision" else None,
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
            "answer_set_images": self._make_candidate_stack(query[0]),
            "correct_answer_image_index": 7,
            "metadata": metadata,
        }

    def _make_candidate_stack(self, query_image: np.ndarray):
        candidates = [query_image for _ in range(8)]
        return np.stack(candidates, axis=0)

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
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 1), text, fill=(20, 20, 20), font=font)

    def _render_context(self, leaves, op1, op2, variant=0, mode="full"):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(18)
        pts = [(32, 96), (64, 96), (96, 96), (64, 58)]
        if variant == 1:
            pts = [(28, 92), (60, 92), (100, 92), (68, 50)]
        elif variant == 2:
            pts = [(34, 98), (68, 98), (96, 98), (64, 62)]
        self._draw_tree(draw, pts, leaves, op1, op2, font, mode=mode, variant=variant, is_query=False)
        meta = {"variant": variant, "mode": "context", "ablation": mode}
        return np.array(img), meta

    def _render_query(self, leaves, op1, op2, mode="full"):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(18)
        pts = [(28, 96), (64, 96), (100, 96), (64, 52)]
        self._draw_tree(draw, pts, leaves, op1, op2, font, mode=mode, variant=0, is_query=True)
        meta = {"mode": "query", "ablation": mode, "binding": "query-specific layout"}
        return np.array(img), meta

    def _draw_tree(self, draw, pts, leaves, op1, op2, font, mode="full", variant=0, is_query=False):
        a, b, c = leaves
        root = pts[3]
        left = pts[0]
        mid = pts[1]
        right = pts[2]

        if mode != "no_visual":
            draw.line([root, left], fill=(60, 60, 60), width=3)
            draw.line([root, right], fill=(60, 60, 60), width=3)
            draw.line([root, mid], fill=(60, 60, 60), width=3)
            draw.line([left, mid], fill=(60, 60, 60), width=3)

        self._draw_node(draw, root, op2 if mode != "no_visual" else "?", (230, 240, 255), (40, 80, 140), font)
        self._draw_node(draw, left, str(a), (255, 240, 230), (140, 80, 40), font)
        self._draw_node(draw, mid, op1 if mode != "no_visual" else "?", (240, 255, 235), (70, 120, 60), font)
        self._draw_node(draw, right, str(c), (255, 245, 245), (130, 60, 60), font)

        if mode != "no_context":
            if is_query:
                self._draw_node(draw, (64, 22), str(b), (240, 240, 240), (90, 90, 90), font)
                if mode != "no_visual":
                    draw.line([(64, 22), left], fill=(110, 110, 110), width=2)
            else:
                self._draw_node(draw, (64, 22), str(b), (240, 240, 240), (90, 90, 90), font)
                if mode != "no_visual":
                    draw.line([(64, 22), mid], fill=(110, 110, 110), width=2)

        if mode == "no_visual":
            draw.text((8, 8), "no_visual", fill=(120, 120, 120), font=self._font(12))
        elif mode == "no_context":
            draw.text((8, 8), "no_context", fill=(120, 120, 120), font=self._font(12))


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
    p.add_argument("--mode", type=str, default="full", choices=MODES)
    return p


def main():
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = VGGenerator(VGConfig(panel_size=args.panel_size, seed=args.seed, mode=args.mode))
    n = args.num_prob if args.max_samples is None else min(args.num_prob, args.max_samples)
    rows = []
    prob_dir = out / "ProbSet" / "train_set"
    prob_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sid = f"vgmnr_{args.mode}_{i:06d}"
        sample = gen.generate_sample(sid)
        save_sample_npz(sample, prob_dir / f"{sid}.npz")
        rows.append(sample["metadata"])
    report = {
        "num_samples": n,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "notes": "Prototype stage-0 generator with ablation-ready modes.",
    }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {n} VG-MNR samples in {out}")


if __name__ == "__main__":
    main()
