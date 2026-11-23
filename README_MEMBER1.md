# Member 1: How to apply this core

## Files you own
- `agent/contracts.py`: shared inputs/outputs and Protocols. Agree on these with all members before they implement their modules.
- `agent/orchestrator.py`: execution order, feedback loop, module calls, error handling, events, report.
- `agent/state.py`: runtime state for one run.
- `agent/stop_policy.py`: stopping rules and best-score tracking.
- `agent/mocks.py`: mock modules so everyone can run and test against the core before real modules exist.
- `agent/ARCHITECTURE.md`: explanation of the agreed design.
- `tests/`, `run_demo.py`: regression tests and a runnable demo.

## Quick start
```bash
python -m pytest        # core tests (stdlib only; pytest needed to run them)
python run_demo.py      # full run with mock modules
```
```python
from agent import Modules, Orchestrator, StopPolicy
report = Orchestrator(Modules(...), StopPolicy(primary_metric="f1")).run(dataset, target)
```

## Recommended workflow
1. Agree on `contracts.py` with the team. Do not let each member invent different input/output shapes.
2. Freeze the method signatures in the Protocols.
3. Each member develops against the core using the mocks, then swaps their own module in.
4. Keep strategy decisions in Member 4's planner. The orchestrator only coordinates and enforces execution/stopping rules.

## Important integration agreements
- `Evaluator.evaluate` must return validation metrics in `ExperimentResult.metrics`, and use the same metric names in `train_metrics` (needed for overfitting diagnosis).
- The configured `primary_metric` must exist in every result; otherwise the iteration fails before anything is logged.
- `Memory.history()` must return the persisted experiment history in chronological order. The diagnoser receives this full history.
- `Memory.has_tried()` / `mark_tried()` receive a signature string computed by the core (SHA-256 of preprocessing, sorted algorithms and tuning). Memory only has to store and look up that string. A plan is marked as tried only after it ran successfully, so a plan that crashed can be retried.
- `Memory.save_best()` should persist the model artifact and metadata, not only the score.
- A `ActionPlan` must list at least one algorithm unless `stop=True`; otherwise the iteration is recorded as an error.
- A preprocessing/training/evaluation exception is recorded as an iteration error; repeated errors trigger StopPolicy.

## Run requirements
Python 3.10+ is recommended. The core itself uses only the Python standard library.
Run from the directory that contains the `agent` package.
