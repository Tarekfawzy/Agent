# Autonomous AI Model Doctor: Member 1 Core Architecture

## Member 1 responsibility
The core orchestrates the workflow, passes agreed contracts between modules,
tracks run state, handles module failures, emits progress events, and applies
stopping rules. It does **not** choose ML strategies; Member 4's DecisionPlanner
owns that responsibility. It does not implement data analysis, preprocessing,
training, evaluation, or persistent storage; the corresponding team members own
those modules.

## Workflow

```text
User dataset + target
        |
        v
DatasetAnalyzer.profile          (Member 2; once)
        |
        v
+---------------- Improvement loop ----------------+
| Memory.history                 (Member 8)          |
| ModelDiagnoser.diagnose         (Member 3)          |
| DecisionPlanner.plan            (Member 4)          |
| Duplicate-plan check + replan  (Member 8)          |
| Preprocessor.run                (Member 5)          |
| Trainer.train                   (Member 6)          |
| Evaluator.evaluate              (Member 7)          |
| Memory.log_experiment/save_best (Member 8)          |
| StopPolicy: continue or stop                        |
+----------------------------------------------------+
        |
        v
AgentReport + progress events
```

## Module ownership and contracts
See `contracts.py`. Each member implements the Protocol assigned to them.
The Orchestrator receives those implementations through `Modules` (dependency
injection), so it does not import Pandas, Scikit-learn, or member-specific code.

## Loop semantics
- `max_iterations=N` runs exactly N iterations. Limit checks happen at the top of each loop; a planner stop (`plan.stop`) is handled inside the iteration and its reason is reported as-is.
- The best result is the strictly best score seen (any gain, however small, is saved via `save_best`).
- `patience` counts **iterations** without progress, not results. Progress means the best score improved by at least `min_improvement`.
- Duplicate plans: the planner is asked for an alternative up to 2 times; if it still repeats, the iteration is skipped and counted (`max_duplicate_plans`). A plan is marked as tried only after it ran successfully.
- `AgentReport.status` is `"completed"` only if at least one model was evaluated; otherwise `"failed"`.

## Stop conditions
`StopPolicy` supports maximum iterations, time budget, target score, patience
(no meaningful improvement), consecutive errors, repeated duplicate plans, and a
stop request from the planner.

## Events for the dashboard (Member 9)
`AgentEvent.event_type` is one of: `started`, `profiled` (issues), `diagnosed` (root cause, problems),
`replanning`, `duplicate_plan`, `planned` (algorithms, preprocessing, tuning, reasoning),
`experiment_completed` (algorithm, metrics, train_metrics, is_best), `iteration_completed`,
`iteration_error`, `stopped`, `failed`. `AgentReport` also carries `decisions` and `elapsed_seconds`.

## Important ML evaluation rule
The `metrics` used by StopPolicy must be validation metrics, not repeated test
set metrics. Keep a final untouched test set for final reporting only. The
Evaluator should use consistent metric names in `metrics` and `train_metrics`.

## Entry point
Construct `Modules(...)`, then `Orchestrator(modules, StopPolicy(...)).run(dataset, target)`.
`agent/mocks.py` provides mock modules (`build_mock_modules()`), `run_demo.py` shows a full run,
and `tests/` holds the regression tests.
