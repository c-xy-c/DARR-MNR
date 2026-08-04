# DARR-MNR Benchmark Status

## Current contract
- 3 context panels
- 8 answer candidates
- label range: 0..7
- primary metric: accuracy
- chance accuracy: 0.125

## Delivered tooling
- `benchmark_protocol.md`: formal dataset and evaluation contract
- `benchmark.py`: validation and scoring CLI
- `benchmark_baseline.py`: random baseline CSV generator
- `tests/test_benchmark.py`: contract tests

## Remaining benchmark work
- official DARR training / inference entrypoint
- official baseline table from paper-grade runs
- ablations and reproducibility runs
- immutable data archive or versioned generation recipe
