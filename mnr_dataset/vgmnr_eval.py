# -*- coding: utf-8 -*-
"""Baseline execution helpers for VG-MNR.

This module provides a small first-round evaluation stack:
- random baseline
- candidate-only heuristic
- number-only heuristic
- small CNN baseline
- lightweight ViT-style baseline

The goal is not to maximize accuracy but to test whether the benchmark
creates a sensible difficulty ladder across the four VG-MNR splits.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


SPLITS = ["full_supervision", "visual_ablation", "context_ablation", "pair_collision"]


@dataclass
class SampleRow:
    sample_id: str
    mode: str
    answer: int
    answer_choices: List[int]
    correct_answer_image_index: int
    context_images: np.ndarray
    answer_set_images: np.ndarray
    metadata: Dict[str, object]


class VGMNRDataset(Dataset):
    def __init__(self, sample_paths: Sequence[Path], split_name: str):
        self.split_name = split_name
        self.rows = [load_sample(path) for path in sample_paths]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "context_images": torch.from_numpy(row.context_images).float().permute(0, 3, 1, 2) / 255.0,
            "answer_set_images": torch.from_numpy(row.answer_set_images).float().permute(0, 3, 1, 2) / 255.0,
            "label": torch.tensor(row.correct_answer_image_index, dtype=torch.long),
        }


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, context_images, answer_set_images):
        b, cnum, ch, h, w = context_images.shape
        contexts = self.backbone(context_images.view(b * cnum, ch, h, w)).flatten(1).view(b, cnum, -1).mean(dim=1)
        b, anum, ch, h, w = answer_set_images.shape
        answers = self.backbone(answer_set_images.view(b * anum, ch, h, w)).flatten(1).view(b, anum, -1)
        ctx = contexts.unsqueeze(1).expand(-1, anum, -1)
        return self.head(torch.cat([ctx, answers], dim=-1)).squeeze(-1)


class TinyViT(nn.Module):
    """A compact ViT-style model with a CNN stem for small-data regimes.

    The CNN stem provides translation-invariant local features that help the
    transformer converge faster on small datasets, while the transformer
    encoder adds global reasoning over the extracted patch tokens.
    """

    def __init__(self, dim=192, heads=6, layers=3):
        super().__init__()
        # CNN stem: 128x128 -> 32x32 with 3 conv + pool stages
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
        )
        # Patch embedding: 32x32 -> 8x8 patches, each 4x4
        self.patch = nn.Conv2d(96, dim, kernel_size=4, stride=4)  # 8x8 = 64 tokens
        # Learned 2D positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, dim))  # 64 patches + 1 CLS
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def encode_image(self, images):
        b, n, c, h, w = images.shape
        x = images.view(b * n, c, h, w)
        x = self.stem(x)  # (b*n, 96, 32, 32)
        x = self.patch(x)  # (b*n, dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (b*n, 64, dim)
        cls = self.cls.expand(x.size(0), -1, -1)  # (b*n, 1, dim)
        x = torch.cat([cls, x], dim=1)  # (b*n, 65, dim)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return x.view(b, n, -1)

    def forward(self, context_images, answer_set_images):
        ctx = self.encode_image(context_images).mean(dim=1)
        ans = self.encode_image(answer_set_images)
        ctx = ctx.unsqueeze(1).expand(-1, ans.size(1), -1)
        return self.head(torch.cat([ctx, ans], dim=-1)).squeeze(-1)


def load_sample(path: Path) -> SampleRow:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(data["metadata_json"].item())
    return SampleRow(
        sample_id=str(metadata["sample_id"]),
        mode=str(metadata["mode"]),
        answer=int(metadata["answer"]),
        answer_choices=list(metadata.get("answer_choices", [])),
        correct_answer_image_index=int(data["correct_answer_image_index"]),
        context_images=data["context_images"],
        answer_set_images=data["answer_set_images"],
        metadata=metadata,
    )


def collect_split_samples(root: Path) -> Dict[str, List[Path]]:
    result = {}
    split_root = root / "splits"
    for split_name in SPLITS:
        split_file = split_root / f"{split_name}.json"
        sample_ids = json.loads(split_file.read_text(encoding="utf-8")) if split_file.exists() else []
        paths = []
        for sample_id in sample_ids:
            matches = list((root / "ProbSet" / "train_set").glob(f"{sample_id}*.npz"))
            if matches:
                paths.append(matches[0])
        result[split_name] = paths
    return result


def random_baseline(rows: Sequence[SampleRow], seed: int = 0) -> float:
    rng = random.Random(seed)
    return sum(int(rng.randrange(8) == row.correct_answer_image_index) for row in rows) / len(rows) if rows else 0.0


def candidate_only_baseline(rows: Sequence[SampleRow]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        if not row.answer_choices:
            continue
        # A simple shortcut heuristic: prefer the central candidate, which is often the baseline-style distractor.
        predicted_idx = min(len(row.answer_choices) // 2, len(row.answer_choices) - 1)
        if predicted_idx == row.correct_answer_image_index:
            correct += 1
    return correct / len(rows)


def _eval_ast(ast):
    if isinstance(ast, int):
        return ast
    left = _eval_ast(ast["left"])
    right = _eval_ast(ast["right"])
    op = ast["op"]
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def number_only_baseline(rows: Sequence[SampleRow]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        lp = row.metadata.get("latent_program")
        if not lp or not row.answer_choices:
            continue
        computed = _eval_ast(lp)
        # Number-only heuristic: choose the exact computed value if present; otherwise choose the closest numeric candidate.
        if computed in row.answer_choices:
            predicted_idx = row.answer_choices.index(computed)
        else:
            predicted_idx = min(range(len(row.answer_choices)), key=lambda i: abs(row.answer_choices[i] - computed))
        if predicted_idx == row.correct_answer_image_index:
            correct += 1
    return correct / len(rows)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        context_images = batch["context_images"].to(device)
        answer_set_images = batch["answer_set_images"].to(device)
        labels = batch["label"].to(device)
        logits = model(context_images, answer_set_images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            context_images = batch["context_images"].to(device)
            answer_set_images = batch["answer_set_images"].to(device)
            labels = batch["label"].to(device)
            logits = model(context_images, answer_set_images)
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += labels.size(0)
    return correct / total if total else 0.0


def run_torch_baseline(model_ctor, split_paths, epochs=8, lr=1e-3, batch_size=8, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for split_name in SPLITS:
        paths = split_paths.get(split_name, [])
        if not paths:
            results[split_name] = {"accuracy": 0.0, "num_samples": 0}
            continue
        dataset = VGMNRDataset(paths, split_name)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = model_ctor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            train_epoch(model, loader, optimizer, device)
        accuracy = eval_model(model, DataLoader(dataset, batch_size=batch_size, shuffle=False), device)
        results[split_name] = {"accuracy": accuracy, "num_samples": len(dataset)}
    return results


def main():
    parser = argparse.ArgumentParser(description="Run first-round VG-MNR baselines.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    split_paths = collect_split_samples(root)
    report = {
        "dataset_root": str(root),
        "split_sizes": {k: len(v) for k, v in split_paths.items()},
        "random": {},
        "candidate_only": {},
        "number_only": {},
        "small_cnn": {},
        "tiny_vit": {},
    }

    for split_name, paths in split_paths.items():
        rows = [load_sample(path) for path in paths]
        report["random"][split_name] = {"accuracy": random_baseline(rows, seed=args.seed), "num_samples": len(rows)}
        report["candidate_only"][split_name] = {"accuracy": candidate_only_baseline(rows), "num_samples": len(rows)}
        report["number_only"][split_name] = {"accuracy": number_only_baseline(rows), "num_samples": len(rows)}

    if torch is not None:
        report["small_cnn"] = run_torch_baseline(SmallCNN, split_paths, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
        report["tiny_vit"] = run_torch_baseline(TinyViT, split_paths, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    else:
        report["small_cnn"] = {split: {"accuracy": None, "num_samples": len(paths)} for split, paths in split_paths.items()}
        report["tiny_vit"] = {split: {"accuracy": None, "num_samples": len(paths)} for split, paths in split_paths.items()}

    report["summary"] = {
        "expected_order": ["Random", "Candidate-only / Number-only", "Small CNN", "Tiny ViT"],
        "interpretation": "If the ordering is weak or inverted, the benchmark likely has shortcut leakage or insufficient split separation.",
    }

    output = Path(args.output) if args.output else root / "vgmnr_baseline_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
