# -*- coding: utf-8 -*-
"""Command line entrypoint for the FVNB-MNR v0 generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnr_dataset.fvnb_baselines import audit_metadata_rows
from mnr_dataset.fvnb_generator import FVNBConfig, FVNBGenerator, RULE_VALUE_SAMPLERS, save_sample_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FVNB-MNR fuzzy-rule samples.")
    parser.add_argument("--num_prob", type=int, default=1000, help="Number of FVNB-MNR samples to generate.")
    parser.add_argument("--output_dir", type=str, default="FVNB-ProbSet", help="Directory for generated .npz files.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic generator seed.")
    parser.add_argument(
        "--rule_schema",
        type=str,
        default="mixed",
        choices=["mixed"] + sorted(RULE_VALUE_SAMPLERS.keys()),
        help="Fuzzy rule schema to generate. Use mixed to sample all schemas.",
    )
    parser.add_argument("--panel_size", type=int, default=FVNBConfig.panel_size, help="Panel size in pixels.")
    parser.add_argument("--theta_pos", type=float, default=0.75, help="Positive truth threshold.")
    parser.add_argument("--delta", type=float, default=0.20, help="Top-1/top-2 truth margin.")
    parser.add_argument("--theta_neg", type=float, default=0.45, help="Diagnostic negative truth threshold.")
    parser.add_argument("--t_norm", type=str, default="product", choices=["product", "minimum", "godel", "min", "lukasiewicz"])
    parser.add_argument("--no_shuffle", action="store_true", help="Keep correct candidate at index 0 for debugging.")
    return parser


def generate_dataset(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = FVNBConfig(
        panel_size=args.panel_size,
        seed=args.seed,
        theta_pos=args.theta_pos,
        delta=args.delta,
        theta_neg=args.theta_neg,
        t_norm=args.t_norm,
        shuffle_candidates=not args.no_shuffle,
    )
    generator = FVNBGenerator(config)
    metadata_path = output_dir / "metadata.jsonl"
    schemas = sorted(RULE_VALUE_SAMPLERS.keys())
    rows = []
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index in range(args.num_prob):
            if args.rule_schema == "mixed":
                rule_id = schemas[index % len(schemas)]
            else:
                rule_id = args.rule_schema
            sample_id = "fvnb_{0:06d}".format(index)
            sample = generator.generate_sample(sample_id, rule_id=rule_id)
            save_sample_npz(sample, output_dir / "{0}.npz".format(sample_id))
            rows.append(sample["metadata"])
            metadata_file.write(json.dumps(sample["metadata"], ensure_ascii=False, sort_keys=True) + "\n")
    report = audit_metadata_rows(rows)
    (output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_prob <= 0:
        parser.error("--num_prob must be positive")
    output_dir = generate_dataset(args)
    print("Generated {0} FVNB-MNR samples in {1}".format(args.num_prob, output_dir))


if __name__ == "__main__":
    main()
