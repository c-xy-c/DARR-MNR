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