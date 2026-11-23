"""Mock modules: stand-ins for the other members' work so the core can be run and tested now.

Replace each mock with the real implementation as teammates deliver it.
"""
from __future__ import annotations

from typing import Optional

from .contracts import (ActionPlan, DataProfile, Diagnosis, ExperimentResult,
                        Issue, Modules, PreparedData, TrainedModel)


class MockAnalyzer:
    def profile(self, dataset, target):
        return DataProfile(100, 5, [Issue("missing_values", "medium", ["age"]),
                                    Issue("class_imbalance", "high", [target])])


class MockDiagnoser:
    def diagnose(self, profile, results):
        if results:
            last = results[-1]
            if last.train_metrics.get("f1", 0) - last.metrics.get("f1", 0) > 0.1:
                return Diagnosis([Issue("overfitting", "high")], "model",
                                 "large gap between train and validation")
        return Diagnosis(list(profile.issues), "data", "data quality issues")


class MockPlanner:
    ORDER = ["logistic_regression", "decision_tree", "random_forest", "svm"]

    def __init__(self):
        self.i = 0

    def plan(self, profile, diagnosis, history):
        if self.i >= len(self.ORDER):
            return ActionPlan(stop=True, stop_reason="No more candidate algorithms")
        algo = self.ORDER[self.i]
        self.i += 1
        return ActionPlan({"imputer": "median", "smote": self.i > 1}, [algo],
                          {"n_trials": 10 * self.i},
                          f"Try {algo} because the root cause looks like: {diagnosis.root_cause}")


class MockPreprocessor:
    def run(self, dataset, target, plan):
        return PreparedData(payload=None, notes={"applied": plan.preprocessing})


class MockTrainer:
    def train(self, data, plan):
        return [TrainedModel(a, object(), plan.tuning) for a in plan.algorithms]


class MockEvaluator:
    """Returns fixed scores so tests are deterministic."""

    def __init__(self, scores: Optional[dict[str, float]] = None):
        self.scores = scores or {"logistic_regression": 0.70, "decision_tree": 0.78,
                                 "random_forest": 0.88, "svm": 0.89}

    def evaluate(self, models, data, plan):
        out = []
        for m in models:
            f1 = self.scores.get(m.name, 0.5)
            out.append(ExperimentResult(m.name, {"f1": f1, "accuracy": f1},
                                        {"f1": min(1.0, f1 + 0.03)}, m.params,
                                        model_ref=m.model))
        return out


class InMemoryMemory:
    def __init__(self):
        self._results: list[ExperimentResult] = []
        self._tried: set[str] = set()
        self.decisions: list = []
        self.best: Optional[ExperimentResult] = None

    def log_experiment(self, result, plan):
        self._results.append(result)
        return f"exp-{len(self._results)}"

    def log_decision(self, iteration, plan, diagnosis):
        self.decisions.append((iteration, plan, diagnosis))

    def has_tried(self, signature): return signature in self._tried
    def mark_tried(self, signature): self._tried.add(signature)
    def history(self): return list(self._results)
    def save_best(self, result): self.best = result


def build_mock_modules(scores: Optional[dict[str, float]] = None) -> Modules:
    return Modules(MockAnalyzer(), MockDiagnoser(), MockPlanner(), MockPreprocessor(),
                   MockTrainer(), MockEvaluator(scores), InMemoryMemory())
