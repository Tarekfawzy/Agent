"""Stopping rules and best-score tracking; this module does not choose ML strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import ExperimentResult
from .state import AgentState


@dataclass
class StopPolicy:
    primary_metric: str = "f1"
    maximize: bool = True
    max_iterations: int = 5
    time_budget_sec: float = 900.0
    target_score: Optional[float] = None
    patience: int = 2                # iterations without meaningful progress before stopping
    min_improvement: float = 0.001   # smallest gain that counts as progress (for patience only)
    max_consecutive_errors: int = 3
    max_duplicate_plans: int = 2     # skipped duplicate plans before stopping

    def score(self, result: ExperimentResult) -> float:
        if self.primary_metric not in result.metrics:
            raise ValueError(
                f"Primary metric '{self.primary_metric}' missing from "
                f"result.metrics; available: {list(result.metrics)}"
            )
        return float(result.metrics[self.primary_metric])

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    def update_best(self, state: AgentState, result: ExperimentResult) -> bool:
        """Track the best result. Any strictly better score becomes the new best,
        however small the gain. Returns True if `result` is the new best."""
        current = self.score(result)
        if state.best_score is None or self._better(current, state.best_score):
            state.best_score = current
            state.best_result = result
            return True
        return False

    def finish_iteration(self, state: AgentState, score_before: Optional[float]) -> bool:
        """Update the patience counter once per iteration (not once per result).
        Progress means the best score moved by at least `min_improvement`.
        Returns True if this iteration made progress."""
        if state.best_score is None:
            progressed = False
        elif score_before is None:
            progressed = True
        else:
            gain = (state.best_score - score_before) if self.maximize \
                else (score_before - state.best_score)
            progressed = gain > 0 and gain >= self.min_improvement
        if progressed:
            state.iterations_without_improvement = 0
        else:
            state.iterations_without_improvement += 1
        return progressed

    def should_stop(self, state: AgentState) -> Optional[str]:
        """Limit checks, evaluated at the top of each loop (planner stops are handled separately)."""
        if state.iteration >= self.max_iterations:
            return "Maximum iteration count reached"
        if state.elapsed_seconds >= self.time_budget_sec:
            return "Time budget reached"
        if state.consecutive_errors >= self.max_consecutive_errors:
            return "Maximum consecutive errors reached"
        if (self.target_score is not None and state.best_score is not None
                and ((state.best_score >= self.target_score) if self.maximize
                     else (state.best_score <= self.target_score))):
            return "Target score reached"
        if state.duplicate_plans >= self.max_duplicate_plans:
            return "Planner keeps proposing already-tried plans"
        if state.iterations_without_improvement >= self.patience:
            return "No sufficient improvement within patience window"
        return None
