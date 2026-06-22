# DARR: A Dual-branch Arithmetic Regression Reasoning Framework for Solving Machine Number Reasoning

This is the official implementation of our AAAI 2025 Oral paper:  
[DARR: A Dual-Branch Arithmetic Regression Reasoning Framework for Solving Machine Number Reasoning](https://ojs.aaai.org/index.php/AAAI/article/view/32127)  
[Chengtai Li](https://scholar.google.com/citations?user=vYL7B1UAAAAJ&hl=en)\*, [Yee Yang Tan](https://scholar.google.com/citations?user=Q0HAjI4AAAAJ&hl=en)\*, [Yuting He](https://scholar.google.com/citations?user=xnNRSj8AAAAJ&hl=en), [Jianfeng Ren](https://research.nottingham.edu.cn/en/persons/jianfeng-ren), [Ruibin Bai](https://research.nottingham.edu.cn/en/persons/ruibin-bai), [Yitian Zhao](https://ytianzhao.github.io/), [Heng Yu](https://research.nottingham.edu.cn/en/persons/heng-yu), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/default.htm)  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2025.  
[[Video](https://underline.io/lecture/113331-darr-a-dual-branch-arithmetic-regression-reasoning-framework-for-solving-machine-number-reasoning)] [[Poster](https://underline.io/lecture/113331-darr-a-dual-branch-arithmetic-regression-reasoning-framework-for-solving-machine-number-reasoning?posterExpanded=true)]

![architecture](figures/model.png)


## Machine Number Reasoning (MNR) Dataset
![architecture](figures/mnr_fig1_2.png)


## Main Results
![result](figures/result.png)


## Requirements
For machine number reasoning (MNR) dataset:
- Python 2.7
- OpenCV
- See `mnr_dataset/requirements.txt` for a detailed list of packages required.

## Experiments
Model training and evaluation code will be released soon.


## Citation
If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{li2025darr,
  title={DARR: A dual-branch arithmetic regression reasoning framework for solving machine number reasoning},
  author={Li, Chengtai and Tan, Yee Yang and He, Yuting and Ren, Jianfeng and Bai, Ruibin and Zhao, Yitian and Yu, Heng and Jiang, Xudong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1373--1382},
  year={2025}
}
```

## Acknowledgement
We sincerely appreciate the following github repos a lot for their valuable code base:
https://github.com/zwh1999anne/Machine-Number-Sense-Dataset


## Python 3 Compatibility Update

This version updates the original `mnr_dataset` generation code to run under Python 3.12.

Main changes include:

- Migrated Python 2 syntax to Python 3 syntax.
- Replaced Python 2-style tuple parameter unpacking in function definitions.
- Updated `range(...)` usages for compatibility with `numpy.random.choice`.
- Fixed integer division issues caused by Python 3’s `/` behavior.
- Ensured array slicing, loop ranges, and index calculations use integer values.
- Fixed OpenCV drawing errors by converting generated coordinates to integers.
- Updated constants such as `CENTER` to avoid float coordinates.
- Verified that the dataset generation script can run successfully under Python 3.12.

This update focuses only on compatibility and does not intentionally change the original dataset generation logic.

## FVNB-MNR Fuzzy Rule Binding Prototype

This repository also includes an experimental FVNB-MNR v0 generator described in `doc/plane.md`. It keeps the original MNR code path untouched and adds a separate fuzzy visual-number binding pipeline under `mnr_dataset/fvnb_*.py`.

Generate a small FVNB-MNR probe set:

```bash
python3 -m mnr_dataset.fvnb_main \
  --num_prob 10 \
  --output_dir FVNB-ProbSet \
  --seed 0 \
  --rule_schema mixed
```

Each generated `.npz` contains:

- `context_images`: shape `(3, H, W)`, with default `H=W=128`
- `answer_set_images`: shape `(8, H, W)`, with default `H=W=128`
- `correct_answer_image_index`: the 8-way label
- `metadata_json`: scene graph, rule schema, raw/normalized role memberships, fuzzy truth scores, hard-role scores, negative types, and shortcut labels

The generator also writes:

- `metadata.jsonl`: one JSON metadata row per sample
- `generation_report.json`: rule counts, negative-type counts, validity checks, fuzzy-oracle accuracy, hard-role accuracy, and hard-role shortcut tie rate

Supported rule schemas:

- `diff_outer_inner_boundary`: `outer - inner = boundary`
- `sum_outer_inner_boundary`: `outer = inner + boundary`
- `ratio_outer_inner_boundary`: `outer / inner = boundary`
- `mixed`: round-robin over all schemas

Supported t-norms:

- `product`
- `min` / `minimum` / `godel`
- `lukasiewicz`

Render one generated sample as an overview PNG:

```bash
python3 -m mnr_dataset.fvnb_visualize FVNB-ProbSet/fvnb_000000.npz \
  --output FVNB-ProbSet/fvnb_000000_grid.png
```

Run the FVNB-MNR tests:

```bash
python3 -m unittest tests.test_fvnb_mnr -v
```

## VQ-Expr 1-5 Calibrated Visual Expression Dataset

This repository also includes a VQ-Expr generator for the newer no-visible-digit design described in `doc/plan.md`. VQ-Expr uses sample-local calibration to decode fuzzy visual attributes into values in `1..5`, compiles visual rule graphs into executable Answer AoTs, and creates 8-way image candidates with single-mutation counterfactual negatives. The presentation view is a RAVEN-style context/choice split: a 1x3 row of complete context panels plus eight full-panel answer candidates. The current visual surface is an A-SIG-lite structured composition: grayscale only, circle quantity entities, one dynamic semantic boundary per panel, explicit four-region in/out binding, no role labels, and RAVEN-inspired configuration families such as `3x3Grid`, `2x2Grid`, `Out-InGrid`, `Out-InCenter`, and `Left-Right`.

Generate a small VQ-Expr probe set:

```bash
python3 -m mnr_dataset.vqexpr_main \
  --num_prob 10 \
  --output_dir VQExpr-ProbSet \
  --seed 0 \
  --rule_schema mixed
```

Each generated `.npz` contains:

- `context_images`: 3 rendered complete context panels for the 1x3 context row.
- `answer_set_images`: 8 rendered image candidates.
- `correct_answer_image_index`: the 8-way label.
- `metadata_json`: calibration context, strip semantics, structured visual surface metadata, per-panel `visual_scene_graph`, dynamic boundary instances, visual quantity family, quantity objects, visual rule graph, Answer AoT, expression schema, candidate mutation logs, scores, and validity checks.

The generator also writes:

- `metadata.jsonl`: one JSON metadata row per sample.
- `generation_report.json`: rule counts, visual-family counts, correct-index counts, candidate visual-stat audit, candidate-only heuristic audit, negative counts, validity counts, and generator settings.
- `vqexpr_000000_overview.png`: a quick visual audit for the first sample.

Current schema:

- `schema_version`: `vqexpr_1_5_avr_1x3_v8_dynamic_boundary`
- `visual_surface.style`: `dynamic_boundary_expression_grayscale_a_sig_lite`
- `visual_surface.scene_graph_schema`: `a_sig_lite_v4`
- visual attribute surfaces: `gray_level`, `size_level`, `count`, `position_set`
- boundary shapes: `circle`, `square`, `diamond`, `hexagon`
- boundary split axes: `horizontal` or `vertical`; every sample binds `q1/q2` to two outer regions and `q3/q4` to two inner regions.

Supported rule schemas:

- `serial`
- `parallel`
- `nested`
- `inverse`
- `calibration`
- `mixed`

Run the VQ-Expr tests:

```bash
python3 -m unittest tests.test_vqexpr -v
```

Run post-hoc VQ-Expr audits on an existing generated set:

```bash
python3 -m mnr_dataset.vqexpr_audit VQExpr-ProbSet \
  --output VQExpr-ProbSet/audit_report.json
```

The post-hoc audit recomputes the metadata oracle, score-argmax oracle, candidate visual statistics, hand-written candidate-only heuristics, and a small learned linear candidate-only probe over visual statistics and answer slot.
