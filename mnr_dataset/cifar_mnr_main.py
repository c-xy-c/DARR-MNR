# -*- coding: utf-8 -*-
"""Generate DARR-MNR panels with per-problem CIFAR-10 digit mappings.

CIFAR-MNR v1 keeps the original DARR-MNR problem logic intact and changes the
visual surface from Arabic numerals to CIFAR-10 image tiles. Each problem has
its own digit-to-CIFAR-class permutation, so the mapping cannot be memorized as
a dataset-global lookup table. Within one problem, the mapping is consistent
across all context panels and answer candidates.
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA_VERSION = "cifar_mnr_v1_per_problem_mapping"


class MaxSamplesReached(RuntimeError):
    pass


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _load_cifar10_with_torchvision(root, seed, allow_synthetic_fallback=False, cifar_url=None):
    if allow_synthetic_fallback:
        return _make_synthetic_class_tiles(seed), "synthetic_fallback"

    try:
        from torchvision.datasets import CIFAR10
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required to load CIFAR-10. Install torchvision or pass "
            "--allow_synthetic_fallback for smoke tests only."
        ) from exc

    if cifar_url:
        CIFAR10.url = cifar_url

    dataset = CIFAR10(root=str(root), train=True, download=True)
    rng = random.Random(seed)
    per_class = {class_id: [] for class_id in range(10)}
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        image, label = dataset[idx]
        if len(per_class[label]) < 256:
            per_class[label].append(image.convert("RGB"))
        if all(per_class[class_id] for class_id in range(10)):
            break

    missing = [class_id for class_id, images in per_class.items() if not images]
    if missing:
        raise RuntimeError("CIFAR-10 loader did not provide samples for classes: %s" % missing)
    return per_class, "cifar10"


def _make_synthetic_class_tiles(seed):
    rng = np.random.default_rng(seed)
    per_class = {}
    palette = np.array(
        [
            [230, 57, 70],
            [29, 53, 87],
            [69, 123, 157],
            [42, 157, 143],
            [233, 196, 106],
            [244, 162, 97],
            [38, 70, 83],
            [131, 56, 236],
            [255, 0, 110],
            [58, 134, 255],
        ],
        dtype=np.uint8,
    )
    for class_id in range(10):
        images = []
        for _ in range(16):
            arr = np.ones((32, 32, 3), dtype=np.uint8) * palette[class_id]
            noise = rng.integers(0, 35, size=(32, 32, 3), dtype=np.uint8)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode="RGB"))
        per_class[class_id] = images
    return per_class


class PerProblemCifarDigitRenderer:
    def __init__(self, cifar_root, seed=0, tile_size=16, grayscale=True, allow_synthetic_fallback=False, cifar_url=None):
        self.seed = seed
        self.tile_size = tile_size
        self.grayscale = grayscale
        self.rng = random.Random(seed)
        self.tiles, self.source = _load_cifar10_with_torchvision(
            cifar_root, seed, allow_synthetic_fallback=allow_synthetic_fallback, cifar_url=cifar_url
        )
        self.next_index = {class_id: 0 for class_id in range(10)}
        self.problem_index = -1
        self.digit_to_class = list(range(10))
        self.class_to_digit = list(range(10))
        self.start_problem()

    def start_problem(self):
        self.problem_index += 1
        self.digit_to_class = list(range(10))
        self.rng.shuffle(self.digit_to_class)
        self.class_to_digit = [None] * 10
        for digit, class_id in enumerate(self.digit_to_class):
            self.class_to_digit[class_id] = digit

    def metadata(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_index": self.problem_index,
            "mapping_policy": "per_problem_permutation",
            "digit_to_cifar_class_id": {str(digit): int(class_id) for digit, class_id in enumerate(self.digit_to_class)},
            "digit_to_cifar_class_name": {
                str(digit): CIFAR10_CLASSES[class_id] for digit, class_id in enumerate(self.digit_to_class)
            },
            "class_id_to_digit": {str(class_id): int(digit) for class_id, digit in enumerate(self.class_to_digit)},
            "class_name_to_digit": {
                CIFAR10_CLASSES[class_id]: int(digit) for class_id, digit in enumerate(self.class_to_digit)
            },
            "mapping_context": {
                "status": "reserved",
                "description": "Future versions can add explicit calibration/context panels that make the per-problem mapping inferable from the problem itself.",
            },
        }

    def sample_class_tile(self, class_id):
        class_id = int(class_id)
        images = self.tiles[class_id]
        idx = self.next_index[class_id] % len(images)
        self.next_index[class_id] += 1
        image = images[idx].resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
        if self.grayscale:
            image = image.convert("L").convert("RGB")
        return image

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


def make_cifar_disp_num(renderer):
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


def _install_cifar_rendering_hooks(renderer, original_main, max_samples=None):
    import mnr_dataset.Drawing as drawing
    import mnr_dataset.Num_Arrange as num_arrange

    saved_count = {"value": 0}

    cifar_disp_num = make_cifar_disp_num(renderer)
    num_arrange.disp_num = cifar_disp_num
    drawing.disp_num = cifar_disp_num

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
                "source_dataset": renderer.source,
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

    dataset_root = Path(args.cifar_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = PerProblemCifarDigitRenderer(
        dataset_root,
        seed=args.seed,
        tile_size=args.tile_size,
        grayscale=not args.rgb_tiles,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        cifar_url=args.cifar_url,
    )

    original_main = _import_original_main_with_args(args.num_prob, args.visualize)
    saved_count = _install_cifar_rendering_hooks(renderer, original_main, max_samples=args.max_samples)

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

    report_path = output_dir / "cifar_mnr_v1_report.txt"
    report_path.write_text(
        "schema_version: %s\n"
        "source: %s\n"
        "cifar_root: %s\n"
        "cifar_url: %s\n"
        "num_prob_per_condition: %s\n"
        "max_samples: %s\n"
        "saved_samples: %s\n"
        "tile_size: %s\n"
        "rgb_tiles: %s\n"
        "mapping_policy: per_problem_permutation\n"
        "cifar_classes: %s\n"
        "note: Every generated .npz contains metadata_json with its local digit-to-CIFAR mapping.\n"
        % (
            SCHEMA_VERSION,
            renderer.source,
            dataset_root,
            args.cifar_url,
            args.num_prob,
            args.max_samples,
            saved_count["value"],
            args.tile_size,
            args.rgb_tiles,
            ", ".join("%d=%s" % (i, name) for i, name in enumerate(CIFAR10_CLASSES)),
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-MNR v1 with a fresh digit-to-CIFAR mapping for each DARR-MNR problem."
    )
    parser.add_argument("--num_prob", type=int, default=1, help="Problems per original DARR-MNR condition.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on saved .npz files for demos and smoke tests.")
    parser.add_argument("--output_dir", type=str, default="CIFAR-MNR-v1", help="Output directory.")
    parser.add_argument(
        "--cifar_root",
        type=str,
        default=str(Path(tempfile.gettempdir()) / "cifar10"),
        help="CIFAR-10 download/cache root.",
    )
    parser.add_argument("--cifar_url", type=str, default=None, help="Optional CIFAR-10 python tarball URL mirror.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tile_size", type=int, default=16, help="Rendered CIFAR tile size in pixels.")
    parser.add_argument("--rgb_tiles", action="store_true", help="Keep CIFAR tiles in RGB before final grayscale conversion.")
    parser.add_argument("--visualize", type=int, choices=[0, 1, 2], default=0, help="Original visualization mode.")
    parser.add_argument(
        "--allow_synthetic_fallback",
        action="store_true",
        help="Use synthetic colored tiles if torchvision/CIFAR is unavailable; for smoke tests only.",
    )
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
