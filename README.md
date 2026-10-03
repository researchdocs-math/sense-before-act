# Risk-Calibrated Sensing for IoT Agents

This package reproduces a calibrated sensing portfolio (CSP), its sensor-query baselines, episode-level risk calibration, fault-stress experiments, and a local language-model integration with emulated IoT tools.

The experiments use a synthetic workload and two public occupancy datasets. Source faults force every report in a group to indicate vacancy. CSP selects between a shared-source planner and a factorized planner using calibration utility among policies whose simultaneous upper risk bound meets the requested target.

## Contents

- `src/`: experiment, calibration, analysis, timing, and local-agent programs.
- `configs/main.json`: the complete main experiment configuration.
- `configs/model.json`: the exact local model repository and revision.
- `data/raw/`: public sensor traces, with attribution in `data/README.md`.
- `results/main/`: candidate calibration records, selected policy tables, selected test outcomes, and fault-stress summaries.
- `results/confirmatory_replications.json`: twenty additional data-partition and simulation replications.
- `results/agent/`: local model outputs, interlock decisions, and timing records.
- `results/analysis.json` and `*_summary.csv`: numeric comparisons.
- `tests/`: mathematical and execution invariants.

## CPU reproduction

Python 3.12 was used for the recorded experiments. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/test_invariants.py
python src/run_experiments.py --workers 8
python src/portfolio.py
python src/replicate_confirmatory.py
python src/analyze.py
python src/timing.py
```

The main run evaluates all seven base methods and 20 cutoffs on 10,000 calibration and 20,000 test episodes for each of three workloads. The portfolio selects from the two planning families; it does not introduce an additional hypothesis test. Tied calibration utility prefers the factorized model, then the smaller cutoff. The twenty confirmation runs use 10,000 calibration and 5,000 test episodes per run. `--workers 8` controls main-run CPU concurrency; the confirmation script uses eight workers. The recorded main run completes in seconds and the confirmation runs in approximately one minute on the stated M3 Pro. Times vary by host. The vectorized replay implementation makes a multi-day training run unnecessary.

The earlier ten-run development sensitivity experiment can also be reproduced with `python src/replicate.py`; its recorded output is `results/replications.json`.

## Expected primary results

At an episode-risk target of 0.01, the selected portfolio has the following results on the 20,000-episode main test sets:

| Workload | Episode risk | Safe completion per request | Mean query cost |
|---|---:|---:|---:|
| Synthetic | 0.004800 | 0.368265 | 2.688975 |
| Office | 0.004400 | 0.326955 | 2.862835 |
| Room | 0.003750 | 0.340535 | 2.990575 |

The configuration, classifier parameters, partition block identities, seeds, candidate risk bounds, and per-episode outcomes are saved. With the pinned CPU dependencies, scalar metrics should agree to numerical precision. Timing and model generation can vary with the host, library versions, and GPU execution.

## Episode semantics and scope

An episode comprises ten requests to enable an energy-saving mode. A permit is unsafe when the occupancy label is positive. A safe completion is a permit on a vacant-room label. Deferral retains the current mode and has zero completion reward. Weighted query costs are resource units, not measured power or communication latency.

Real data are split into disjoint 100-row blocks separated by 20-row gaps. Fitting, development, and replay receive approximately 40%, 20%, and 40% of the blocks. A replay episode draws a block and a ten-row window. Calibration and test use independent draws from the same fixed replay bank and may reuse original rows. Guarantees are conditional on this bank and the specified fault simulator. Faults persist across all requests in an episode; the sensing belief resets for each request. The experiments are observational replay, not closed-loop physical building control.

All policies can select the always-defer option. The main confidence correction is `0.05 / 420`; the twenty confirmation runs use `0.05 / 8400`. The same simultaneous bounds support both main risk targets. Fault-stress outcomes are not used for selection.

## Local GPU agent experiment

The recorded integration used Apple M3 Pro, macOS 14.8.1, 18 GB unified memory, MLX 0.32.3, MLX-LM 0.32.0, and the four-bit Qwen2.5-1.5B-Instruct conversion pinned in `configs/model.json`.

On an Apple Silicon host:

```bash
python -m pip install -r requirements-agent.txt
python src/download_model.py --destination models/qwen2.5-1.5b-4bit
TOKENIZERS_PARALLELISM=false python src/run_agent.py --model-path models/qwen2.5-1.5b-4bit --batch-size 8
```

The model uses greedy decoding and a 32-token output limit. The tool interface is an emulator: `verify` executes the selected sensing table, `apply` is allowed only after a permit, and `defer` retains the emulated state. The 120-request integration includes 24 attempts to override verification. Complete generated responses and tool traces are saved. A separate set of twelve serial two-call interactions measures latency. CPU policy timing excludes sensor acquisition and language inference.

## Reproduction verification

```bash
python src/verify_results.py
```

This checks the primary expected metrics, saved calibration selection, scalar versus vectorized routing, and recorded model/interlock traces. `MANIFEST.sha256` records the packaged source, data, configuration, and result files.

## License and citation

The implementation is distributed under the MIT License. The two UCI datasets retain their Creative Commons Attribution 4.0 license and are credited in `data/README.md`. The downloaded Qwen model has its own Apache 2.0 license. `CITATION.cff` provides the software citation.
