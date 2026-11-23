import pytest

from agent import (ActionPlan, ExperimentResult, Modules, Orchestrator,
                   StopPolicy, TrainedModel)
from agent.mocks import InMemoryMemory, build_mock_modules


def run(policy=None, modules=None, scores=None):
    events = []
    mods = modules or build_mock_modules(scores)
    orch = Orchestrator(mods, policy or StopPolicy(max_iterations=10, patience=10), events.append)
    return orch.run(None, "y"), events, mods


class SeqPlanner:
    """Always proposes a new, unique plan with the given algorithms."""
    def __init__(self, algorithms=("a",)):
        self.n = 0
        self.algorithms = list(algorithms)

    def plan(self, profile, diagnosis, history):
        self.n += 1
        return ActionPlan({"i": self.n}, self.algorithms, {}, f"plan {self.n}")


class ScriptedEvaluator:
    """evaluate() returns the next list of {algorithm: f1} from the script."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def evaluate(self, models, data, plan):
        scores = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [ExperimentResult(m.name, {"f1": scores.get(m.name, 0.0)}) for m in models]


def with_modules(planner=None, evaluator=None, trainer=None):
    mods = build_mock_modules()
    if planner is not None:
        mods.planner = planner
    if evaluator is not None:
        mods.evaluator = evaluator
    if trainer is not None:
        mods.trainer = trainer
    return mods


# ---------------------------------------------------------------- happy path
def test_full_run_picks_best_and_reports():
    report, events, mods = run()
    assert report.status == "completed"
    assert report.best_result.algorithm == "svm"
    assert mods.memory.best.algorithm == "svm"
    assert report.elapsed_seconds >= 0 and report.decisions
    kinds = {e.event_type for e in events}
    assert {"started", "profiled", "diagnosed", "planned", "experiment_completed",
            "iteration_completed", "stopped"} <= kinds


def test_planned_event_carries_reasoning_for_the_ui():
    _, events, _ = run()
    planned = [e for e in events if e.event_type == "planned"]
    assert planned and all(e.data["reasoning"] for e in planned)


def test_stops_on_target_score():
    report, _, _ = run(StopPolicy(max_iterations=10, patience=10, target_score=0.85))
    assert report.stop_reason == "Target score reached" and report.iterations == 3


def test_planner_stop_reason_is_reported():
    report, _, _ = run()
    assert report.stop_reason == "No more candidate algorithms"


# ----------------------------------------- regression: the six reviewed bugs
@pytest.mark.parametrize("n", [1, 2, 5])
def test_max_iterations_runs_exactly_n_experiments(n):          # bug 1
    ev = ScriptedEvaluator([{"a": 0.5 + 0.01 * i} for i in range(20)])
    report, _, _ = run(StopPolicy(max_iterations=n, patience=99),
                       with_modules(SeqPlanner(), ev))
    assert ev.calls == n and report.iterations == n
    assert report.stop_reason == "Maximum iteration count reached"


def test_patience_counts_iterations_not_results():              # bug 2
    ev = ScriptedEvaluator([{"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.5},
                            {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.5},
                            {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.5}])
    report, _, _ = run(StopPolicy(max_iterations=10, patience=2),
                       with_modules(SeqPlanner("abcd"), ev))
    # iter 1 improves; iter 2 and 3 do not -> stops after 3 iterations, not 1
    assert report.iterations == 3
    assert report.stop_reason == "No sufficient improvement within patience window"


def test_tiny_improvement_still_becomes_best():                 # bug 3
    ev = ScriptedEvaluator([{"a": 0.8000}, {"a": 0.8005}])
    report, _, mods = run(StopPolicy(max_iterations=2, patience=99, min_improvement=0.001),
                          with_modules(SeqPlanner(), ev))
    assert report.best_result.metrics["f1"] == 0.8005
    assert mods.memory.best.metrics["f1"] == 0.8005


def test_tiny_improvement_does_not_reset_patience():
    ev = ScriptedEvaluator([{"a": 0.8000}, {"a": 0.8002}, {"a": 0.8004}])
    report, _, _ = run(StopPolicy(max_iterations=10, patience=2, min_improvement=0.001),
                       with_modules(SeqPlanner(), ev))
    assert report.stop_reason == "No sufficient improvement within patience window"
    assert report.iterations == 3


class Boom:
    def train(self, data, plan):
        raise ValueError("boom")


def test_status_failed_when_no_model_was_ever_produced():       # bug 4
    report, events, _ = run(modules=with_modules(SeqPlanner(), ScriptedEvaluator([{"a": 1}]), Boom()))
    assert report.status == "failed" and report.best_result is None
    assert report.errors and events[-1].event_type == "failed"


def test_failed_plan_can_be_retried_and_planner_stop_wins():    # bug 5
    class SamePlanner:
        def plan(self, profile, diagnosis, history):
            return ActionPlan({"x": 1}, ["a"], {}, "same plan")

    class FlakyTrainer:
        calls = 0
        def train(self, data, plan):
            FlakyTrainer.calls += 1
            if FlakyTrainer.calls == 1:
                raise ValueError("transient")
            return [TrainedModel("a", object())]

    mods = with_modules(SamePlanner(), ScriptedEvaluator([{"a": 0.7}]), FlakyTrainer())
    report, _, _ = run(StopPolicy(max_iterations=2, patience=99), mods)
    assert report.status == "completed" and report.best_result.metrics["f1"] == 0.7

    class StopTwice:
        def plan(self, profile, diagnosis, history):
            return ActionPlan(stop=True, stop_reason="done")
    mods = with_modules(StopTwice())
    mods.memory.mark_tried(Orchestrator._plan_signature(ActionPlan(stop=True)))
    report, _, _ = run(modules=mods)
    assert report.stop_reason == "done"


def test_duplicate_plan_triggers_replan_then_skip():
    class RepeatsOnce:
        def __init__(self): self.calls = 0
        def plan(self, profile, diagnosis, history):
            self.calls += 1
            algo = "a" if self.calls <= 2 else "b"
            return ActionPlan({}, [algo], {}, "x")
    planner = RepeatsOnce()
    mods = with_modules(planner, ScriptedEvaluator([{"a": 0.6, "b": 0.7}]))
    report, events, _ = run(StopPolicy(max_iterations=2, patience=99), mods)
    assert {r.algorithm for r in report.history} == {"a", "b"}
    assert any(e.event_type == "replanning" for e in events)


def test_always_duplicate_planner_is_stopped():
    class Same:
        def plan(self, profile, diagnosis, history):
            return ActionPlan({}, ["a"], {}, "x")
    report, _, _ = run(StopPolicy(max_iterations=20, patience=99),
                       with_modules(Same(), ScriptedEvaluator([{"a": 0.6}])))
    assert len(report.history) == 1
    assert report.stop_reason == "Planner keeps proposing already-tried plans"


def test_signature_ignores_algorithm_order():                   # bug 6
    a = Orchestrator._plan_signature(ActionPlan({}, ["svm", "knn"], {}))
    b = Orchestrator._plan_signature(ActionPlan({}, ["knn", "svm"], {}))
    assert a == b


# ------------------------------------------------------------ extra safeguards
def test_empty_plan_without_stop_is_an_error_not_a_silent_run():
    class Empty:
        def plan(self, profile, diagnosis, history):
            return ActionPlan()
    report, _, _ = run(StopPolicy(max_iterations=10, patience=10, max_consecutive_errors=2),
                       with_modules(Empty()))
    assert report.status == "failed"
    assert "no algorithms" in report.errors[0]
    assert report.stop_reason == "Maximum consecutive errors reached"


def test_missing_primary_metric_fails_before_touching_memory():
    class NoF1:
        def evaluate(self, models, data, plan):
            return [ExperimentResult(m.name, {"accuracy": 0.9}) for m in models]
    mods = with_modules(SeqPlanner(), NoF1())
    report, _, _ = run(StopPolicy(max_iterations=3, patience=99, max_consecutive_errors=1), mods)
    assert report.status == "failed" and mods.memory.history() == []


def test_profiler_failure_fails_fast():
    class Bad:
        def profile(self, d, t):
            raise OSError("bad csv")
    mods = build_mock_modules()
    mods.analyzer = Bad()
    report, events, _ = run(modules=mods)
    assert report.status == "failed" and report.iterations == 0
    assert events[-1].event_type == "failed"


def test_broken_event_callback_does_not_break_the_run():
    def bad(event):
        raise RuntimeError("ui crashed")
    orch = Orchestrator(build_mock_modules(), StopPolicy(max_iterations=3, patience=9), bad)
    assert orch.run(None, "y").status == "completed"


def test_minimize_mode():
    ev = ScriptedEvaluator([{"a": 0.5}, {"a": 0.3}, {"a": 0.4}])
    report, _, _ = run(StopPolicy(primary_metric="f1", maximize=False, max_iterations=3, patience=9),
                       with_modules(SeqPlanner(), ev))
    assert report.best_result.metrics["f1"] == 0.3
