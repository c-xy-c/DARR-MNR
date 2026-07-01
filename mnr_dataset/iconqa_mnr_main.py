# -*- coding: utf-8 -*-
"""Generate DARR-MNR panels with IconQA/Icon645 icons replacing digits.

IconQA itself is a QA benchmark, so this generator uses Icon645, the official
IconQA companion icon-classification dataset, as the visual token source. Each
DARR-MNR problem gets a fresh digit-to-icon-class permutation, and the mapping
is stored in metadata_json.
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = "icon645_mnr_v1_per_problem_mapping"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class MaxSamplesReached(RuntimeError):
    pass


def _make_synthetic_icon_tiles(seed, num_classes=10):
    rng = np.random.default_rng(seed)
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (38, 70, 83),
        (131, 56, 236),
        (255, 0, 110),
        (58, 134, 255),
    ]
    shapes = ["circle", "square", "triangle", "diamond", "star"]
    per_class = {}
    class_names = []
    for class_id in range(num_classes):
        class_name = "synthetic_icon_%02d" % class_id
        class_names.append(class_name)
        images = []
        for variant in range(16):
            image = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            color = palette[class_id % len(palette)]
            jitter = int(rng.integers(-12, 13))
            fill = tuple(max(0, min(255, channel + jitter)) for channel in color) + (255,)
            outline = (20, 20, 20, 255)
            shape = shapes[(class_id + variant) % len(shapes)]
            margin = int(rng.integers(8, 15))
            box = [margin, margin, 64 - margin, 64 - margin]
            if shape == "circle":
                draw.ellipse(box, fill=fill, outline=outline, width=3)
            elif shape == "square":
                draw.rectangle(box, fill=fill, outline=outline, width=3)
            elif shape == "triangle":
                draw.polygon([(32, margin), (64 - margin, 64 - margin), (margin, 64 - margin)], fill=fill, outline=outline)
            elif shape == "diamond":
                draw.polygon([(32, margin), (64 - margin, 32), (32, 64 - margin), (margin, 32)], fill=fill, outline=outline)
            else:
                points = [(32, margin), (39, 25), (56, 25), (43, 37), (48, 54), (32, 44), (16, 54), (21, 37), (8, 25), (25, 25)]
                draw.polygon(points, fill=fill, outline=outline)
            images.append(image)
        per_class[class_id] = images
    return per_class, class_names, "synthetic_icon645_fallback"


def _resolve_icon_root(icon645_root):
    root = Path(icon645_root).resolve()
    candidates = [
        root / "colored_icons_final",
        root / "icon645" / "colored_icons_final",
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
            if class_dirs:
                return candidate
    raise FileNotFoundError(
        "Cannot find Icon645 class folders. Expected one of: %s"
        % ", ".join(str(candidate) for candidate in candidates)
    )


def _load_icon645(icon645_root, seed, selected_classes=None, num_visual_tokens=10, max_images_per_class=256, allow_synthetic_fallback=False):
    if allow_synthetic_fallback:
        return _make_synthetic_icon_tiles(seed, num_classes=max(10, num_visual_tokens))

    icon_root = _resolve_icon_root(icon645_root)
    rng = random.Random(seed)
    class_dirs = sorted([path for path in icon_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    if selected_classes:
        selected_set = set(selected_classes)
        class_dirs = [path for path in class_dirs if path.name in selected_set]
        missing = sorted(selected_set - {path.name for path in class_dirs})
        if missing:
            raise ValueError("Selected Icon645 classes not found: %s" % missing)
    else:
        rng.shuffle(class_dirs)
        class_dirs = sorted(class_dirs[:num_visual_tokens], key=lambda path: path.name)

    if len(class_dirs) < num_visual_tokens:
        raise ValueError("Need at least %d icon classes, found %d" % (num_visual_tokens, len(class_dirs)))

    class_dirs = class_dirs[:num_visual_tokens]
    per_class = {}
    class_names = []
    for class_id, class_dir in enumerate(class_dirs):
        files = [path for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            raise ValueError("Icon class has no image files: %s" % class_dir)
        rng.shuffle(files)
        images = []
        for path in files[:max_images_per_class]:
            try:
                images.append(Image.open(path).convert("RGBA"))
            except Exception:
                continue
        if not images:
            raise ValueError("Icon class images could not be loaded: %s" % class_dir)
        per_class[class_id] = images
        class_names.append(class_dir.name)

    return per_class, class_names, "icon645"


class PerProblemIconDigitRenderer:
    def __init__(
        self,
        icon645_root,
        seed=0,
        tile_size=24,
        grayscale=False,
        selected_classes=None,
        num_visual_tokens=10,
        allow_synthetic_fallback=False,
    ):
        self.seed = seed
        self.tile_size = tile_size
        self.grayscale = grayscale
        self.rng = random.Random(seed)
        self.tiles, self.class_names, self.source = _load_icon645(
            icon645_root,
            seed,
            selected_classes=selected_classes,
            num_visual_tokens=num_visual_tokens,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        self.next_index = {class_id: 0 for class_id in range(len(self.class_names))}
        self.problem_index = -1
        self.digit_to_class = list(range(10))
        self.class_to_digit = list(range(10))
        self.start_problem()

    def start_problem(self):
        self.problem_index += 1
        self.digit_to_class = list(range(10))
        self.rng.shuffle(self.digit_to_class)
        self.class_to_digit = [None] * len(self.class_names)
        for digit, class_id in enumerate(self.digit_to_class):
            self.class_to_digit[class_id] = digit

    def metadata(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_index": self.problem_index,
            "source_dataset": self.source,
            "mapping_policy": "per_problem_permutation",
            "digit_to_icon_class_id": {str(digit): int(class_id) for digit, class_id in enumerate(self.digit_to_class)},
            "digit_to_icon_class_name": {
                str(digit): self.class_names[class_id] for digit, class_id in enumerate(self.digit_to_class)
            },
            "class_id_to_digit": {str(class_id): int(digit) for class_id, digit in enumerate(self.class_to_digit)},
            "class_name_to_digit": {
                self.class_names[class_id]: int(digit) for class_id, digit in enumerate(self.class_to_digit)
            },
            "mapping_context": {
                "status": "reserved",
                "description": "IconQA itself is a QA benchmark; Icon645 icons are used here as visual digit tokens. Future versions can add explicit calibration panels if a fully inferable local mapping is required.",
            },
        }

    def sample_class_tile(self, class_id):
        class_id = int(class_id)
        images = self.tiles[class_id]
        idx = self.next_index[class_id] % len(images)
        self.next_index[class_id] += 1
        image = images[idx].resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (self.tile_size, self.tile_size), (255, 255, 255, 0))
        canvas.alpha_composite(image)
        if self.grayscale:
            return canvas.convert("L").convert("RGB")
        return canvas.convert("RGB")

    def draw_number(self, image, value, x, y):
        digit_chars = list(str(int(value)))
        total_width = len(digit_chars) * self.tile_size + max(0, len(digit_chars) - 1) * 2
        left = int(round(x - total_width / 2.0))
        top = int(round(y - self.tile_size / 2.0))
        for offset, digit_char in enumerate(digit_chars):
            digit = int(digit_char)
            class_id = self.digit_to_class[digit]
            tile = self.sample_class_tile(class_id)
            image.paste(tile, (left + offset * (self.tile_size + 2), top))


def make_icon_disp_num(renderer):
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


def _install_icon_rendering_hooks(renderer, original_main, max_samples=None):
    import mnr_dataset.Drawing as drawing
    import mnr_dataset.Num_Arrange as num_arrange

    saved_count = {"value": 0}
    icon_disp_num = make_icon_disp_num(renderer)
    num_arrange.disp_num = icon_disp_num
    drawing.disp_num = icon_disp_num

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
    selected_classes = args.icon_classes.split(",") if args.icon_classes else None

    renderer = PerProblemIconDigitRenderer(
        args.icon645_root,
        seed=args.seed,
        tile_size=args.tile_size,
        grayscale=args.grayscale,
        selected_classes=selected_classes,
        num_visual_tokens=10,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )

    original_main = _import_original_main_with_args(args.num_prob, args.visualize)
    saved_count = _install_icon_rendering_hooks(renderer, original_main, max_samples=args.max_samples)

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

    report_path = output_dir / "icon645_mnr_v1_report.txt"
    report_path.write_text(
        "schema_version: %s\n"
        "source: %s\n"
        "icon645_root: %s\n"
        "num_prob_per_condition: %s\n"
        "max_samples: %s\n"
        "saved_samples: %s\n"
        "tile_size: %s\n"
        "grayscale_tiles: %s\n"
        "mapping_policy: per_problem_permutation\n"
        "icon_classes: %s\n"
        "note: IconQA is not used directly because it is a complete QA benchmark; Icon645 is the official IconQA companion icon dataset used as visual tokens.\n"
        % (
            SCHEMA_VERSION,
            renderer.source,
            Path(args.icon645_root).resolve(),
            args.num_prob,
            args.max_samples,
            saved_count["value"],
            args.tile_size,
            args.grayscale,
            ", ".join(renderer.class_names),
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Icon645-MNR by replacing DARR-MNR digits with IconQA/Icon645 visual tokens.")
    parser.add_argument("--num_prob", type=int, default=1, help="Problems per original DARR-MNR condition.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on saved .npz files for demos and smoke tests.")
    parser.add_argument("--output_dir", type=str, default="Icon645-MNR-v1", help="Output directory.")
    parser.add_argument(
        "--icon645_root",
        type=str,
        default=str(Path(tempfile.gettempdir()) / "icon645"),
        help="Icon645 root. Expected colored_icons_final/<class>/*.png or icon645/colored_icons_final/<class>/*.png.",
    )
    parser.add_argument("--icon_classes", type=str, default=None, help="Optional comma-separated Icon645 class names to use as the 10 visual tokens.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tile_size", type=int, default=24, help="Rendered icon tile size in pixels.")
    parser.add_argument("--grayscale", action="store_true", help="Convert icon tiles and output panels to grayscale.")
    parser.add_argument("--visualize", type=int, choices=[0, 1, 2], default=0, help="Original visualization mode.")
    parser.add_argument("--allow_synthetic_fallback", action="store_true", help="Use synthetic icon tiles when Icon645 is unavailable; for smoke tests only.")
    args = parser.parse_args()
    if args.num_prob <= 0:
        raise ValueError("--num_prob must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive when provided")
    if args.tile_size <= 0:
        raise ValueError("--tile_size must be positive")
    if args.icon_classes and len(args.icon_classes.split(",")) < 10:
        raise ValueError("--icon_classes must contain at least 10 comma-separated class names")
    return args


if __name__ == "__main__":
    run_generation(parse_args())
