# -*- coding: utf-8 -*-
"""Generate DARR-MNR panels with CLEVR-style 3D rendered tokens.

This version does not rely on external 3D assets. It synthesizes a small set of
CLEVR-like 3D primitives (sphere, cube, cylinder) with controlled colors,
lighting, shading, and perspective cues, then uses them as digit tokens.

Each DARR-MNR problem gets a fresh digit-to-token permutation and the mapping is
stored in metadata_json.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = "clevr3d_mnr_v1_per_problem_mapping"


class MaxSamplesReached(RuntimeError):
    pass


TOKEN_SPECS = [
    {"shape": "sphere", "color": (232, 76, 61), "name": "red_sphere"},
    {"shape": "sphere", "color": (65, 105, 225), "name": "blue_sphere"},
    {"shape": "cube", "color": (71, 179, 125), "name": "green_cube"},
    {"shape": "cube", "color": (240, 194, 73), "name": "yellow_cube"},
    {"shape": "cylinder", "color": (155, 89, 182), "name": "purple_cylinder"},
    {"shape": "cylinder", "color": (52, 152, 219), "name": "cyan_cylinder"},
    {"shape": "sphere", "color": (242, 153, 74), "name": "orange_sphere"},
    {"shape": "cube", "color": (219, 112, 147), "name": "pink_cube"},
    {"shape": "cylinder", "color": (141, 110, 99), "name": "brown_cylinder"},
    {"shape": "cube", "color": (78, 205, 196), "name": "teal_cube"},
]


class CLEVR3DTokenRenderer:
    def __init__(self, seed=0, tile_size=24, grayscale=False):
        self.seed = seed
        self.tile_size = tile_size
        self.grayscale = grayscale
        self.rng = random.Random(seed)
        self.token_specs = list(TOKEN_SPECS)
        self.next_index = {i: 0 for i in range(len(self.token_specs))}
        self.problem_index = -1
        self.digit_to_token = list(range(10))
        self.token_to_digit = list(range(10))
        self.start_problem()

    def start_problem(self):
        self.problem_index += 1
        self.digit_to_token = list(range(10))
        self.rng.shuffle(self.digit_to_token)
        self.token_to_digit = [None] * len(self.token_specs)
        for digit, token_id in enumerate(self.digit_to_token):
            self.token_to_digit[token_id] = digit

    def metadata(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_index": self.problem_index,
            "source_dataset": "synthetic_clevr3d",
            "mapping_policy": "per_problem_permutation",
            "digit_to_token_id": {str(digit): int(token_id) for digit, token_id in enumerate(self.digit_to_token)},
            "digit_to_token_name": {
                str(digit): self.token_specs[token_id]["name"] for digit, token_id in enumerate(self.digit_to_token)
            },
            "token_id_to_digit": {str(token_id): int(digit) for token_id, digit in enumerate(self.token_to_digit)},
            "token_name_to_digit": {
                self.token_specs[token_id]["name"]: int(digit) for token_id, digit in enumerate(self.token_to_digit)
            },
            "mapping_context": {
                "status": "reserved",
                "description": "CLEVR-style 3D primitives are used as digit tokens. Future versions may add explicit calibration clues if inferable local mappings are needed.",
            },
            "token_specs": self.token_specs,
        }

    def sample_token(self, token_id):
        token_id = int(token_id)
        spec = self.token_specs[token_id]
        idx = self.next_index[token_id]
        self.next_index[token_id] = idx + 1
        return self._render_token(spec, idx)

    def draw_number(self, image, value, x, y):
        digit_chars = list(str(int(value)))
        total_width = len(digit_chars) * self.tile_size + max(0, len(digit_chars) - 1) * 2
        left = int(round(x - total_width / 2.0))
        top = int(round(y - self.tile_size / 2.0))
        for offset, digit_char in enumerate(digit_chars):
            digit = int(digit_char)
            token_id = self.digit_to_token[digit]
            tile = self.sample_token(token_id)
            image.paste(tile, (left + offset * (self.tile_size + 2), top), tile if tile.mode == "RGBA" else None)

    def _render_token(self, spec, variant_index):
        size = 64
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        shape = spec["shape"]
        color = spec["color"]
        if shape == "sphere":
            self._draw_sphere(canvas, color, variant_index)
        elif shape == "cube":
            self._draw_cube(canvas, color, variant_index)
        elif shape == "cylinder":
            self._draw_cylinder(canvas, color, variant_index)
        else:
            raise ValueError("Unsupported shape: %s" % shape)
        if self.tile_size != size:
            canvas = canvas.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        if self.grayscale:
            return canvas.convert("L").convert("RGB")
        return canvas.convert("RGB")

    @staticmethod
    def _clamp_color(color, factor):
        return tuple(max(0, min(255, int(round(channel * factor)))) for channel in color)

    def _draw_shadow(self, canvas, center_x, center_y, rx, ry, alpha=90):
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        draw.ellipse([center_x - rx, center_y - ry, center_x + rx, center_y + ry], fill=(0, 0, 0, alpha))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(shadow)

    def _draw_sphere(self, canvas, color, variant_index):
        from PIL import ImageFilter

        size = canvas.size[0]
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        cx = size * 0.5 + (variant_index % 3 - 1) * 1.5
        cy = size * 0.52 + ((variant_index // 3) % 3 - 1) * 1.0
        r = size * 0.30 + (variant_index % 2) * 0.6
        yy, xx = np.mgrid[0:size, 0:size]
        dx = (xx - cx) / r
        dy = (yy - cy) / r
        dist2 = dx * dx + dy * dy
        mask = dist2 <= 1.0
        z = np.zeros_like(dist2)
        z[mask] = np.sqrt(np.clip(1.0 - dist2[mask], 0.0, 1.0))
        light = np.array([-0.45, -0.35, 1.0], dtype=np.float32)
        light = light / np.linalg.norm(light)
        nx = dx
        ny = dy
        nz = z
        lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
        base = np.array(color, dtype=np.float32)
        factor = 0.45 + 0.55 * lambert
        rgb = np.clip(base[None, None, :] * factor[..., None], 0, 255)
        specular_center = np.exp(-(((xx - (cx - r * 0.28)) ** 2 + (yy - (cy - r * 0.28)) ** 2) / (2 * (r * 0.18) ** 2)))
        rgb = np.clip(rgb + specular_center[..., None] * 55, 0, 255)
        arr[..., :3][mask] = rgb[mask].astype(np.uint8)
        arr[..., 3][mask] = 255
        sphere = Image.fromarray(arr, mode="RGBA")
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        draw.ellipse([cx - r * 0.78, cy + r * 0.60, cx + r * 0.78, cy + r * 0.90], fill=(0, 0, 0, 90))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(shadow)
        canvas.alpha_composite(sphere)

    def _draw_cube(self, canvas, color, variant_index):
        from PIL import ImageFilter

        size = canvas.size[0]
        cx = size * 0.50 + (variant_index % 3 - 1) * 1.4
        cy = size * 0.48 + ((variant_index // 3) % 3 - 1) * 1.0
        s = size * 0.28
        dx = s * 0.55
        dy = s * 0.38
        front = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
        top = [(cx - s, cy - s), (cx, cy - s - dy), (cx + s, cy - s), (cx, cy - s + dy)]
        side = [(cx + s, cy - s), (cx + s + dx, cy - s + dy), (cx + s + dx, cy + s + dy), (cx + s, cy + s)]
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        draw.ellipse([cx - s * 0.85, cy + s * 0.82, cx + s * 0.95, cy + s * 1.18], fill=(0, 0, 0, 80))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(shadow)
        body = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(body)
        front_fill = self._clamp_color(color, 0.95) + (255,)
        top_fill = self._clamp_color(color, 1.18) + (255,)
        side_fill = self._clamp_color(color, 0.72) + (255,)
        edge = (20, 20, 20, 255)
        draw.polygon(top, fill=top_fill, outline=edge)
        draw.polygon(side, fill=side_fill, outline=edge)
        draw.polygon(front, fill=front_fill, outline=edge)
        draw.line([front[0], top[1], side[1]], fill=(255, 255, 255, 120), width=2)
        canvas.alpha_composite(body)

    def _draw_cylinder(self, canvas, color, variant_index):
        from PIL import ImageFilter

        size = canvas.size[0]
        cx = size * 0.50 + (variant_index % 3 - 1) * 1.2
        cy = size * 0.49 + ((variant_index // 3) % 3 - 1) * 1.0
        rx = size * 0.30
        ry = size * 0.16
        h = size * 0.42
        top = [cx - rx, cy - h * 0.5 - ry, cx + rx, cy - h * 0.5 + ry]
        bottom = [cx - rx, cy + h * 0.5 - ry, cx + rx, cy + h * 0.5 + ry]
        body_left = cx - rx
        body_top = cy - h * 0.5
        body_right = cx + rx
        body_bottom = cy + h * 0.5
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        draw.ellipse([cx - rx * 0.9, cy + h * 0.48, cx + rx * 0.9, cy + h * 0.80], fill=(0, 0, 0, 80))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(shadow)
        body = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(body)
        edge = (20, 20, 20, 255)
        for i in range(int(body_left), int(body_right)):
            t = (i - body_left) / max(1.0, body_right - body_left)
            shade = 0.68 + 0.34 * (1.0 - abs(t - 0.45) * 1.6)
            col = self._clamp_color(color, shade) + (255,)
            draw.line([(i, body_top), (i, body_bottom)], fill=col)
        draw.ellipse(top, fill=self._clamp_color(color, 1.12) + (255,), outline=edge, width=2)
        draw.ellipse(bottom, fill=self._clamp_color(color, 0.82) + (255,), outline=edge, width=2)
        draw.rectangle([body_left, body_top, body_right, body_bottom], outline=edge, width=2)
        highlight = [(cx - rx * 0.45, body_top + ry * 0.2), (cx - rx * 0.20, body_top + h * 0.18)]
        draw.line(highlight, fill=(255, 255, 255, 110), width=2)
        canvas.alpha_composite(body)


def make_clevr_disp_num(renderer):
    from mnr_dataset.Num_Arrange import mutate_constant

    def disp_num(img, int_list, pos_list, blank, font_size, show_center, chosen_mutate_type=None):
        num_pos = len(pos_list)
        answer = 0
        image = Image.fromarray(img).convert("RGB")

        mark_positions = []
        if chosen_mutate_type == "MutFalse":
            num_to_mutate = np.random.choice([1, 2])
            marks = np.random.choice(list(range(0, num_pos)), size=num_to_mutate, replace=False)
            mark_positions = marks.tolist()

        for i in range(0, num_pos):
            x, y = pos_list[i]
            value = int_list[i]
            if i in mark_positions:
                value = mutate_constant(value)
                answer = int_list[i]
            renderer.draw_number(image, value, x, y)

        if renderer.grayscale:
            image = image.convert("L")
        return answer, image, -1

    return disp_num


def _import_original_main_with_args(num_prob, visualize):
    old_argv = sys.argv[:]
    sys.argv = ["main.py", "--num_prob", str(num_prob), "--visualize", str(visualize)]
    try:
        import mnr_dataset.main as original_main
    finally:
        sys.argv = old_argv
    return original_main


def _install_rendering_hooks(renderer, original_main, max_samples=None):
    import mnr_dataset.Drawing as drawing
    import mnr_dataset.Num_Arrange as num_arrange

    saved_count = {"value": 0}
    clevr_disp_num = make_clevr_disp_num(renderer)
    num_arrange.disp_num = clevr_disp_num
    drawing.disp_num = clevr_disp_num

    original_drawing_panels = drawing.drawing_panels

    def drawing_panels_with_problem_mapping(*args, **kwargs):
        renderer.start_problem()
        return original_drawing_panels(*args, **kwargs)

    drawing.drawing_panels = drawing_panels_with_problem_mapping
    original_main.drawing_panels = drawing_panels_with_problem_mapping

    original_savez = original_main.np.savez

    def savez_with_mapping_metadata(file, *args, **kwargs):
        if max_samples is not None and saved_count["value"] >= max_samples:
            raise MaxSamplesReached()
        metadata = dict(renderer.metadata())
        metadata.update(
            {
                "tile_size": renderer.tile_size,
                "grayscale_tiles": renderer.grayscale,
            }
        )
        kwargs.setdefault("metadata_json", json.dumps(metadata, ensure_ascii=False))
        original_savez(file, *args, **kwargs)
        saved_count["value"] += 1

    original_main.np.savez = savez_with_mapping_metadata
    return saved_count


def run_generation(args):
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parents[0]
    for import_path in (str(repo_root), str(module_dir)):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = CLEVR3DTokenRenderer(seed=args.seed, tile_size=args.tile_size, grayscale=args.grayscale)
    original_main = _import_original_main_with_args(args.num_prob, args.visualize)
    saved_count = _install_rendering_hooks(renderer, original_main, max_samples=args.max_samples)

    old_cwd = os.getcwd()
    os.chdir(str(output_dir))
    try:
        original_main.np.random.seed(args.seed)
        original_main.random.seed(args.seed)
        original_main.NUM_PROB = args.num_prob
        original_main.VISUALIZE = args.visualize
        try:
            original_main.main()
        except MaxSamplesReached:
            pass
    finally:
        os.chdir(old_cwd)

    report_path = output_dir / "clevr3d_mnr_v1_report.txt"
    report_path.write_text(
        "schema_version: %s\n"
        "source: %s\n"
        "num_prob_per_condition: %s\n"
        "max_samples: %s\n"
        "saved_samples: %s\n"
        "tile_size: %s\n"
        "grayscale_tiles: %s\n"
        "mapping_policy: per_problem_permutation\n"
        "token_names: %s\n"
        "note: This is a CLEVR-style synthetic 3D token dataset designed to replace CIFAR-like symbols in DARR-MNR.\n"
        % (
            SCHEMA_VERSION,
            "synthetic_clevr3d",
            args.num_prob,
            args.max_samples,
            saved_count["value"],
            args.tile_size,
            args.grayscale,
            ", ".join(spec["name"] for spec in TOKEN_SPECS),
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CLEVR-style 3D token MNR samples.")
    parser.add_argument("--num_prob", type=int, default=1, help="Problems per original DARR-MNR condition.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on saved .npz files for demos and smoke tests.")
    parser.add_argument("--output_dir", type=str, default="CLEVR3D-MNR-v1", help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tile_size", type=int, default=24, help="Rendered token tile size in pixels.")
    parser.add_argument("--grayscale", action="store_true", help="Convert rendered tokens and panels to grayscale.")
    parser.add_argument("--visualize", type=int, choices=[0, 1, 2], default=0, help="Original visualization mode.")
    args = parser.parse_args()
    if args.num_prob <= 0:
        raise ValueError("--num_prob must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive when provided")
    if args.tile_size <= 0:
        raise ValueError("--tile_size must be positive")
    return args


if __name__ == "__main__":
    run_generation(parse_args())
