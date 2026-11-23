"""Shared contracts between the orchestrator and team modules.

The core intentionally does not import pandas or scikit-learn. Team modules
may use those libraries internally, but must return these shared objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Issue:
    kind: str
    severity: str = "medium"  # low | medium | high
    columns: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Diagnosis:
    problems: list[Issue] = field(default_factory=list)
    root_cause: str = "none"  # data | features | model | none | unknown
    explanation: str = ""


@dataclass
class ActionPlan:
    preprocessing: dict[str, Any] = field(default_factory=dict)
    algorithms: list[str] = field(default_factory=list)  # must be non-empty unless stop=True
    tuning: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""  # why the agent chose this plan (shown in the dashboard)
    stop: bool = False
    stop_reason: str = ""


@dataclass
class PreparedData:
    # Expected to contain train/validation splits, not a test set used for tuning.
    payload: Any
    pipeline: Any = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainedModel:
    name: str
    model: Any
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    algorithm: str
    metrics: dict[str, float]  # validation metrics used for model selection
    train_metrics: dict[str, float] = field(default_factory=dict)  # same metric names as `metrics`
    params: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    model_ref: Any = None
    iteration: int = 0
    experiment_id: Optional[str] = None


@dataclass
class AgentEvent:
    """Progress event for the UI.

    event_type is one of: started, profiled, diagnosed, replanning, duplicate_plan,
    planned, experiment_completed, iteration_completed, iteration_error, stopped, failed.
    """
    event_type: str
    iteration: int
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReport:
    status: str  # "completed" (a best model exists) | "failed"
    stop_reason: str
    iterations: int
    best_result: Optional[ExperimentResult]
    profile: Optional[DataProfile]
    last_diagnosis: Optional[Diagnosis]
    errors: list[str] = field(default_factory=list)
    history: list[ExperimentResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class DatasetAnalyzer(Protocol):
    def profile(self, dataset: Any, target: str) -> DataProfile: ...


@runtime_checkable
class ModelDiagnoser(Protocol):
    def diagnose(self, profile: DataProfile,
                 results: list[ExperimentResult]) -> Diagnosis: ...


@runtime_checkable
class DecisionPlanner(Protocol):
    def plan(self, profile: DataProfile, diagnosis: Diagnosis,
             history: list[ExperimentResult]) -> ActionPlan: ...


@runtime_checkable
class Preprocessor(Protocol):
    def run(self, dataset: Any, target: str, plan: ActionPlan) -> PreparedData: ...


@runtime_checkable
class Trainer(Protocol):
    def train(self, data: PreparedData, plan: ActionPlan) -> list[TrainedModel]: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, models: list[TrainedModel], data: PreparedData,
                 plan: ActionPlan) -> list[ExperimentResult]: ...


@runtime_checkable
class Memory(Protocol):
    def log_experiment(self, result: ExperimentResult, plan: ActionPlan) -> str: ...
    def log_decision(self, iteration: int, plan: ActionPlan, diagnosis: Diagnosis) -> None: ...
    def has_tried(self, signature: str) -> bool: ...
    def mark_tried(self, signature: str) -> None: ...
    def history(self) -> list[ExperimentResult]: ...  # chronological order
    def save_best(self, result: ExperimentResult) -> None: ...


@dataclass
class Modules:
    analyzer: DatasetAnalyzer
    diagnoser: ModelDiagnoser
    planner: DecisionPlanner
    preprocessor: Preprocessor
    trainer: Trainer
    evaluator: Evaluator
    memory: Memory
