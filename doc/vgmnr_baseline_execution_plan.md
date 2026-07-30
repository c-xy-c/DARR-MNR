# VG-MNR Baseline Execution Plan

Date: 2026-07-30

This document turns the current VG-MNR prototype into a concrete first-round evaluation protocol. It is intentionally limited to the present stable dataset baseline and a small set of algorithmic baselines that can reveal whether the benchmark is meaningful before any further dataset changes.

It assumes the current stable VG-MNR prototype state:

- `full`, `no_visual`, `no_context`, and `collision` all generate stably in small batches.
- `full` maps to `full_supervision`.
- `no_visual` maps to `visual_ablation`.
- `no_context` maps to `context_ablation`.
- `collision` maps to `pair_collision`.
- The stable collision baseline is `operand_swap`.

## 1. Goal of the first-round experiment

The purpose is not to maximize accuracy. The purpose is to test whether the benchmark behaves as designed.

We want to know:

1. Can the dataset be generated reproducibly in a fixed version?
2. Do the proposed algorithm classes form a sensible performance ladder?
3. Do the four splits produce the expected difficulty ordering?
4. Are there obvious shortcuts that make the benchmark invalid?

## 2. Data version to freeze

Before running baselines, freeze one dataset version and do not change generation rules during the baseline run.

Recommended frozen validation set:

- `full_supervision`
- `visual_ablation`
- `context_ablation`
- `pair_collision`

Use one fixed seed per split and keep the generated outputs under versioned directories.

## 3. Baseline suite

The first-round suite contains five concrete baselines.

### 3.1 Random baseline

**Input**: no structured reasoning, uniform random 8-way choice.

**Purpose**:

- Establish the lower bound.
- Confirm the label space behaves as expected.

**Expectation**:

- Accuracy should be close to chance.
- It should not show a strong split preference.

### 3.2 Candidate-only heuristic

**Input**: only the eight answer candidates, no context/query reasoning.

**Purpose**:

- Detect candidate bias.
- Test whether the answer slot or rendering leaks the label.

**Expectation**:

- Close to random or only slightly above it.
- If substantially above chance, candidate construction is too biased.

### 3.3 Number-only heuristic

**Input**: the numeric or symbol-level summary extracted from the sample, without using the full visual reasoning path.

**Purpose**:

- Test whether the task can be solved through shallow numeric shortcuts.
- Probe whether the benchmark really depends on the intended visual reasoning path.

**Expectation**:

- Meaningfully below the real model baseline.
- Should degrade on ablations and collision.

### 3.4 Small CNN baseline

**Input**: rendered images with a lightweight convolutional encoder.

**Purpose**:

- Provide a simple learnable visual baseline.
- Show whether a basic image model can extract any useful signal.

**Expectation**:

- Better than random and simple heuristics.
- Still clearly below a stronger multimodal baseline.

### 3.5 ViT-style / multimodal baseline

**Input**: a stronger image encoder or multimodal model.

**Purpose**:

- Serve as the strong reference baseline.
- Test whether the benchmark separates stronger and weaker models.

**Expectation**:

- Best or near-best on `full`.
- Clear drops on `no_visual`, `no_context`, and `collision`.

## 4. Why these five baselines

This combination gives three levels of evidence:

1. **Chance level**
   - Random baseline

2. **Shortcut level**
   - Candidate-only heuristic
   - Number-only heuristic

3. **Learned reasoning level**
   - Small CNN
   - ViT-style / multimodal model

This is enough for the first validation round. More algorithms can be added later, but these five already test whether the benchmark is doing the right job.

## 5. Expected result ordering

A healthy benchmark should roughly show:

```text
Random < Candidate-only / Number-only < Small CNN < ViT-style / multimodal
```

This is not a formal theorem, but it is the expected shape if the dataset is meaningful.

## 6. Split-level expectations

### 6.1 `full_supervision`

- Strong baselines should do best here.
- Heuristics should not approach the strong model.

### 6.2 `visual_ablation`

- Performance should drop relative to `full_supervision`.
- If it does not drop, visual necessity is too weak.

### 6.3 `context_ablation`

- Performance should also drop relative to `full_supervision`.
- If it does not drop, context is not contributing enough.

### 6.4 `pair_collision`

- Should be harder than `full_supervision` or at least comparable in difficulty.
- If it is too easy, the collision family is not meaningful.
- If every model collapses to random, the collision family is too hard or too ambiguous.

## 7. What counts as a good first-round outcome

A good first-round outcome is not "everyone scores high".
It is a pattern where:

- Random stays low.
- Candidate-only / Number-only stay low to moderate.
- Small CNN improves over the shortcut baselines.
- ViT-style / multimodal is best overall.
- `full_supervision` is easiest.
- `visual_ablation` and `context_ablation` are harder than `full_supervision`.
- `pair_collision` is hardest or at least clearly non-trivial.

## 8. What counts as a bad first-round outcome

### 8.1 Shortcut failure

If candidate-only or number-only is too strong, then the dataset is leaking answer information.

Action:

- Rework candidate construction.
- Reduce positional leakage.
- Re-check whether metadata or rendering exposes the answer.

### 8.2 No split separation

If all splits look similar, the benchmark is not testing the intended factors.

Action:

- Strengthen visual necessity.
- Strengthen context necessity.
- Tighten or redesign the split logic.

### 8.3 Collision too easy

If `pair_collision` is close to `full_supervision`, collision is not useful.

Action:

- Add stronger collision families later.
- Introduce more difficult paired interventions.

### 8.4 Everything is random-like

If even the strongest model is near chance, the task is too hard or too under-specified.

Action:

- Simplify the reasoning path.
- Reduce ambiguity.
- Increase the clarity of the answer construction.

## 9. Recommended execution order

1. Freeze a versioned validation set.
2. Run Random baseline.
3. Run Candidate-only heuristic.
4. Run Number-only heuristic.
5. Run Small CNN.
6. Run ViT-style / multimodal baseline.
7. Compare split-level results.
8. Decide whether the dataset needs revision.

## 10. Reporting template

For each baseline, report:

- Overall accuracy.
- Accuracy on each split.
- Candidate-only accuracy if relevant.
- Collision pair consistency if relevant.
- Common error patterns.
- Whether the result matches the expected ranking.

## 11. Decision rule for dataset revision

Revise the dataset if any of the following is true:

- Shortcut baselines are too strong.
- Split-level differences are absent or inverted.
- Collision is not harder than full supervision.
- The strongest model does not outperform weaker baselines.
- The error distribution suggests label leakage or overly ambiguous construction.

If none of the above is true, keep the dataset fixed and move to the next round of baseline analysis.
