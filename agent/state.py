"""Runtime state owned by the orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Optional

from .contracts import DataProfile, Diagnosis, ExperimentResult


@dataclass
class AgentState:
    started_at: float = field(default_factory=monotonic)
    iteration: int = 0
    profile: Optional[DataProfile] = None
    last_diagnosis: Optional[Diagnosis] = None
    best_result: Optional[ExperimentResult] = None
    best_score: Optional[float] = None
    iterations_without_improvement: int = 0
    consecutive_errors: int = 0
    duplicate_plans: int = 0
    errors: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at
