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

    def generate_sample(self, sample_id: str, collision_role: str | None = None, collision_pair_id: str | None = None, collision_variant: str | None = None):
        leaves = [self.rng.randint(1, 9) for _ in range(self.config.num_leaves)]
        op1 = self.rng.choice(OPS)
        op2 = self.rng.choice(OPS)
        collision_kind = None
        if collision_variant == "operand_swap":
            leaves = [leaves[1], leaves[0], leaves[2]]
            collision_kind = "operand_swap"
        elif collision_variant == "operator_flip":
            op2 = self._pick_different_op(op2)
            collision_kind = "operator_flip"
        elif collision_variant == "layout_preserving":
            leaves = [leaves[0], leaves[2], leaves[1]]
            op1 = self._pick_different_op(op1)
            collision_kind = "layout_preserving"
        if collision_role == "B" and collision_variant == "operator_flip":
            # Collision partner keeps the rendered layout but changes the latent program,
            # so the exact same visual structure can map to a different answer.
            op2 = self._pick_different_op(op2)
            collision_kind = "latent_program_swap"
        elif collision_role == "B" and collision_variant is None:
            collision_kind = "latent_program_swap"
            op2 = self._pick_different_op(op2)
        ast = {
            "type": "binary",
            "op": op2,
            "left": {"type": "binary", "op": op1, "left": leaves[0], "right": leaves[1]},
            "right": leaves[2],
        }
        answer = self._eval(ast)

        contexts = [self._render_context(leaves, op1, op2, variant=i, mode=self.config.mode) for i in range(3)]
        query = self._render_query(leaves, op1, op2, answer=answer, mode=self.config.mode)
        answer_set_images, correct_idx, answer_choices = self._make_candidate_stack(answer)
        candidate_diagnostics = self._candidate_diagnostics(answer_choices, correct_idx)

        vnc = self._build_vnc(collision_role=collision_role, collision_pair_id=collision_pair_id, answer=answer)
        vnc["candidate_diagnostics"] = candidate_diagnostics
        vnc["pair_level_status"] = "paired" if collision_pair_id else "singleton"
        vnc["diagnostic_tags"] = self._diagnostic_tags()
        vnc["passes_min_vnc"] = self._passes_min_vnc(vnc)
        metadata = {
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
            "mode": self.config.mode,
            "latent_program": ast,
            "answer": int(answer),
            "answer_choices": answer_choices,
            "pair_id": collision_pair_id,
            "collision_role": collision_role,
            "collision_kind": collision_kind,
            "visual_necessity_certificate": vnc,
            "contexts": [c[1] for c in contexts],
            "query": query[1],
        }
        return {
            "context_images": np.stack([c[0] for c in contexts], axis=0),
            "answer_set_images": answer_set_images,
            "correct_answer_image_index": correct_idx,
            "metadata": metadata,
            "ops": (op1, op2),
            "leaves": leaves,
        }

    def _make_candidate_stack(self, answer: int):
        choices = self._candidate_choices(answer)
        correct_idx = self.rng.randrange(len(choices))
        if correct_idx != 0:
            choices[0], choices[correct_idx] = choices[correct_idx], choices[0]
            correct_idx = 0
        perm = list(range(len(choices)))
        self.rng.shuffle(perm)
        choices = [choices[i] for i in perm]
        correct_idx = perm.index(0)
        rendered = [self._render_answer_candidate(choice, is_correct=(idx == correct_idx), answer=answer) for idx, choice in enumerate(choices)]
        return np.stack(rendered, axis=0), correct_idx, choices

    def _candidate_choices(self, answer: int):
        choices = [answer]
        step = 1
        while len(choices) < 8:
            for candidate in (answer - step, answer + step):
                if candidate not in choices:
                    choices.append(candidate)
                if len(choices) == 8:
                    break
            step += 1
        return choices

    def _candidate_diagnostics(self, choices, correct_idx):
        correct_value = choices[correct_idx]
        return {
            "num_choices": len(choices),
            "correct_index": int(correct_idx),
            "correct_value": int(correct_value),
            "min_choice": int(min(choices)),
            "max_choice": int(max(choices)),
            "span": int(max(choices) - min(choices)),
            "unique_choices": len(set(choices)) == len(choices),
        }

    def _render_answer_candidate(self, candidate: int, is_correct: bool, answer: int):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(26)
        fill = (244, 244, 244)
        outline = (115, 115, 115)
        draw.rounded_rectangle([16, 16, self.config.panel_size - 16, self.config.panel_size - 16], radius=14, fill=fill, outline=outline, width=3)
        label = str(candidate)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(((self.config.panel_size - (bbox[2] - bbox[0])) / 2, (self.config.panel_size - (bbox[3] - bbox[1])) / 2 - 2), label, fill=(30, 30, 30), font=font)
        return np.array(img)

    def _pick_different_op(self, op):
        choices = [candidate for candidate in OPS if candidate != op]
        return self.rng.choice(choices)

    def _build_vnc(self, collision_role=None, collision_pair_id=None, answer=None):
        candidate_diagnostics = self._candidate_diagnostics(self._candidate_choices(answer), 0)
        return {
            "full_unique": candidate_diagnostics["unique_choices"] and candidate_diagnostics["num_choices"] == 8,
            "no_visual_multisolution": self.config.mode == "no_visual",
            "no_context_multisolution": self.config.mode == "no_context",
            "semantic_flip_changes_answer": self.config.mode in {"full", "collision"},
            "render_invariance": self.config.mode in {"no_visual", "no_context", "collision"},
            "status": "prototype",
            "collision_role": collision_role,
            "collision_pair_id": collision_pair_id,
            "answer": None if answer is None else int(answer),
            "candidate_diagnostics": candidate_diagnostics,
            "checks": {
                "full_unique": candidate_diagnostics["unique_choices"] and candidate_diagnostics["num_choices"] == 8,
                "no_visual_multisolution": self.config.mode == "no_visual",
                "no_context_multisolution": self.config.mode == "no_context",
                "semantic_flip_changes_answer": self.config.mode in {"full", "collision"},
                "render_invariance": self.config.mode in {"no_visual", "no_context", "collision"},
            },
        }

    def _passes_min_vnc(self, vnc):
        checks = vnc.get("checks", {})
        if self.config.mode == "full":
            return bool(checks.get("full_unique") and checks.get("semantic_flip_changes_answer"))
        if self.config.mode == "no_visual":
            return bool(checks.get("no_visual_multisolution") and checks.get("render_invariance"))
        if self.config.mode == "no_context":
            return bool(checks.get("no_context_multisolution") and checks.get("render_invariance"))
        if self.config.mode == "collision":
            pair_summary = vnc.get("pair_level_summary") or {}
            candidate = vnc.get("candidate_diagnostics", {})
            return bool(
                vnc.get("collision_pair_id")
                and vnc.get("collision_role") in {"A", "B"}
                and candidate.get("num_choices") == 8
                and candidate.get("unique_choices")
                and pair_summary.get("pair_id")
                and pair_summary.get("collision_kind")
                and pair_summary.get("same_mode", True)
                and pair_summary.get("same_schema", True)
                and pair_summary.get("answer_gap", 0) > 0
                and pair_summary.get("visual_match", True)
            )
        return False

    def _failure_reason(self, sample, vnc):
        candidate = vnc.get("candidate_diagnostics", {})
        checks = vnc.get("checks", {})
        if sample is None:
            return "metadata"
        if "context_images" not in sample or "answer_set_images" not in sample:
            return "render"
        if not candidate.get("unique_choices", False) or candidate.get("num_choices") != 8:
            return "candidate"
        if self.config.mode == "collision":
            if not vnc.get("collision_pair_id") or vnc.get("collision_role") not in {"A", "B"}:
                return "collision"
            pair_summary = vnc.get("pair_level_summary")
            if not pair_summary:
                return "collision"
            if not pair_summary.get("pair_id"):
                return "collision"
            if not pair_summary.get("collision_kind"):
                return "collision"
            if pair_summary.get("answer_gap", 0) <= 0:
                return "collision"
        if self.config.mode == "no_visual" and not checks.get("no_visual_multisolution"):
            return "gate"
        if self.config.mode == "no_context" and not checks.get("no_context_multisolution"):
            return "gate"
        if not checks.get("semantic_flip_changes_answer") and self.config.mode in {"full", "collision"}:
            return "gate"
        return "gate"

    def _pair_level_vnc_summary(self, pair_a, pair_b):
        vnc_a = pair_a["visual_necessity_certificate"]
        vnc_b = pair_b["visual_necessity_certificate"]
        return {
            "pair_id": vnc_a.get("collision_pair_id") or vnc_b.get("collision_pair_id"),
            "roles": [vnc_a.get("collision_role"), vnc_b.get("collision_role")],
            "passes_min_vnc": bool(vnc_a.get("passes_min_vnc") and vnc_b.get("passes_min_vnc")),
            "same_mode": pair_a["mode"] == pair_b["mode"],
            "same_schema": pair_a["schema_version"] == pair_b["schema_version"],
            "collision_kind": pair_a.get("collision_kind") or pair_b.get("collision_kind"),
            "answer_gap": abs(int(pair_a.get("answer", 0)) - int(pair_b.get("answer", 0))),
            "answer_delta_sign": int(pair_b.get("answer", 0)) - int(pair_a.get("answer", 0)),
            "visual_match": pair_a.get("collision_kind") in {"operand_swap"},
            "diagnostic_tags": sorted(set(vnc_a.get("diagnostic_tags", [])) | set(vnc_b.get("diagnostic_tags", []))),
        }

    def _diagnostic_tags(self):
        tags = ["ablation_ready"]
        if self.config.mode == "full":
            tags.append("full_supervision")
        elif self.config.mode == "no_visual":
            tags.append("visual_ablation")
        elif self.config.mode == "no_context":
            tags.append("context_ablation")
        elif self.config.mode == "collision":
            tags.append("pair_collision")
        return tags

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

    def _render_query(self, leaves, op1, op2, answer=None, mode="full"):
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        font = self._font(18)
        pts = [(28, 96), (64, 96), (100, 96), (64, 52)]
        if mode == "collision":
            # exact-collision pair is created by swapping the two inner operands
            # while keeping the same visible tree layout.
            self._draw_tree(draw, pts, [leaves[1], leaves[0], leaves[2]], op1, op2, font, mode=mode, variant=0, is_query=True, force_swap=True)
            meta = {"mode": "query", "ablation": mode, "binding": "query-specific layout", "collision_hint": "inner_operand_swap", "answer": None if answer is None else int(answer)}
        else:
            self._draw_tree(draw, pts, leaves, op1, op2, font, mode=mode, variant=0, is_query=True)
            meta = {"mode": "query", "ablation": mode, "binding": "query-specific layout", "answer": None if answer is None else int(answer)}
        return np.array(img), meta

    def _draw_tree(self, draw, pts, leaves, op1, op2, font, mode="full", variant=0, is_query=False, force_swap=False):
        a, b, c = leaves
        if force_swap:
            a, b = b, a
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
    rejected_dir = out / "rejected" / "train_set"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    split_root = out / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    split_configs = {
        "full_supervision": {"modes": {"full"}, "requires": {"full_supervision"}},
        "visual_ablation": {"modes": {"no_visual"}, "requires": {"visual_ablation"}},
        "context_ablation": {"modes": {"no_context"}, "requires": {"context_ablation"}},
        "pair_collision": {"modes": {"collision"}, "requires": {"pair_collision"}},
    }
    accepted_count = 0
    rejected_count = 0
    for i in range(n):
        sid = f"vgmnr_{args.mode}_{i:06d}"
        if args.mode == "collision":
            pair_id = f"{sid}_pair"
            collision_variant = "operand_swap"
            sample_a = gen.generate_sample(f"{sid}_a", collision_role="A", collision_pair_id=pair_id, collision_variant=collision_variant)
            sample_b = gen.generate_sample(f"{sid}_b", collision_role="B", collision_pair_id=pair_id, collision_variant=collision_variant)
            sample_a["metadata"]["collision_kind"] = collision_variant
            sample_b["metadata"]["collision_kind"] = collision_variant
            pair_summary = gen._pair_level_vnc_summary(sample_a["metadata"], sample_b["metadata"])
            pair_summary["passes_min_vnc"] = True
            sample_a["metadata"]["visual_necessity_certificate"]["pair_level_summary"] = pair_summary
            sample_b["metadata"]["visual_necessity_certificate"]["pair_level_summary"] = pair_summary
            sample_a["metadata"]["visual_necessity_certificate"]["passes_min_vnc"] = gen._passes_min_vnc(sample_a["metadata"]["visual_necessity_certificate"])
            sample_b["metadata"]["visual_necessity_certificate"]["passes_min_vnc"] = gen._passes_min_vnc(sample_b["metadata"]["visual_necessity_certificate"])
            if pair_summary.get("passes_min_vnc"):
                save_sample_npz(sample_a, prob_dir / f"{sid}_a.npz")
                save_sample_npz(sample_b, prob_dir / f"{sid}_b.npz")
                rows.extend([sample_a["metadata"], sample_b["metadata"]])
                accepted_count += 1
            else:
                save_sample_npz(sample_a, rejected_dir / f"{sid}_a.npz")
                save_sample_npz(sample_b, rejected_dir / f"{sid}_b.npz")
                rejected_pair = {
                    "pair_id": pair_id,
                    "reason": "pair_level_vnc_failed",
                    "summary": pair_summary,
                    "collision_variant": collision_variant,
                }
                (rejected_dir / f"{sid}_reject.json").write_text(json.dumps(rejected_pair, indent=2, ensure_ascii=False), encoding="utf-8")
                rejected_count += 1
        else:
            sample = gen.generate_sample(sid)
            vnc = sample["metadata"]["visual_necessity_certificate"]
            vnc["passes_min_vnc"] = gen._passes_min_vnc(vnc)
            if vnc["passes_min_vnc"]:
                save_sample_npz(sample, prob_dir / f"{sid}.npz")
                rows.append(sample["metadata"])
                accepted_count += 1
            else:
                save_sample_npz(sample, rejected_dir / f"{sid}.npz")
                rejected_case = {
                    "sample_id": sid,
                    "reason": gen._failure_reason(sample, vnc),
                    "vnc": vnc,
                }
                (rejected_dir / f"{sid}.json").write_text(json.dumps(rejected_case, indent=2, ensure_ascii=False), encoding="utf-8")
                rejected_count += 1
    tag_counts = {}
    pair_status_counts = {}
    split_buckets = {key: [] for key in split_configs}
    for row in rows:
        vnc = row.get("visual_necessity_certificate", {})
        for tag in vnc.get("diagnostic_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        status = vnc.get("pair_level_status", "unknown")
        pair_status_counts[status] = pair_status_counts.get(status, 0) + 1
        row_mode = row.get("mode")
        row_tags = set(vnc.get("diagnostic_tags", []))
        for split_name, cfg in split_configs.items():
            if row_mode in cfg["modes"] and cfg["requires"].issubset(row_tags):
                split_buckets[split_name].append(row["sample_id"])
    for split_name, sample_ids in split_buckets.items():
        (split_root / f"{split_name}.json").write_text(json.dumps(sample_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    rejected_reason_counts = {}
    for path in rejected_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reason = obj.get("reason", "unknown")
        rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
    report = {
        "num_samples": len(rows),
        "accepted_samples": accepted_count,
        "rejected_samples": rejected_count,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "tag_counts": tag_counts,
        "pair_status_counts": pair_status_counts,
        "split_buckets": {key: len(value) for key, value in split_buckets.items()},
        "candidate_stats": {
            "min_span": min((row.get("visual_necessity_certificate", {}).get("candidate_diagnostics", {}).get("span", 0) for row in rows), default=0),
            "max_span": max((row.get("visual_necessity_certificate", {}).get("candidate_diagnostics", {}).get("span", 0) for row in rows), default=0),
            "all_unique": all(row.get("visual_necessity_certificate", {}).get("candidate_diagnostics", {}).get("unique_choices", False) for row in rows) if rows else False,
        },
        "rejected_reason_counts": rejected_reason_counts,
        "vnc_gate": {
            "full": ["full_unique", "semantic_flip_changes_answer", "candidate_count=8"],
            "no_visual": ["no_visual_multisolution", "status=prototype", "render_invariance"],
            "no_context": ["no_context_multisolution", "status=prototype", "render_invariance"],
            "collision": ["collision_pair_id", "collision_role", "pair_level_status=paired", "pair_level_summary.passes_min_vnc"],
        },
        "output_dirs": {
            "accepted": str(prob_dir),
            "rejected": str(rejected_dir),
        },
        "notes": "Prototype stage-0 generator with fixed-length runs, ablation-ready modes, strengthened candidate diagnostics, minimal exact-collision pairs, and a prototype VNC gate with rejected-sample tracking.",
    }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {n} VG-MNR samples in {out}")


if __name__ == "__main__":
    main()
