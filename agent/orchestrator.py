"""Coordinates team modules and runs the autonomous improvement loop.

DecisionPlanner (Member 4) chooses *what* to try. This class controls *when*
and *in what order* modules run, validates results, tracks state and stops.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Optional

from .contracts import ActionPlan, AgentEvent, AgentReport, Modules
from .state import AgentState
from .stop_policy import StopPolicy

logger = logging.getLogger(__name__)
EventCallback = Callable[[AgentEvent], None]

MAX_REPLAN_ATTEMPTS = 2  # times we ask the planner for an alternative to an already-tried plan


class Orchestrator:
    def __init__(self, modules: Modules, stop_policy: Optional[StopPolicy] = None,
                 on_event: Optional[EventCallback] = None) -> None:
        self.modules = modules
        self.stop_policy = stop_policy or StopPolicy()
        self.on_event = on_event

    # ------------------------------------------------------------------ utils
    def _emit(self, event_type: str, state: AgentState, message: str,
              data: Optional[dict[str, Any]] = None) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(AgentEvent(event_type, state.iteration, message, data or {}))
        except Exception:
            # UI/reporting callbacks must not break the ML workflow.
            logger.exception("Agent event callback failed")

    @staticmethod
    def _plan_signature(plan: ActionPlan) -> str:
        """Deterministic fingerprint of a plan; algorithm order does not matter."""
        raw = json.dumps({
            "preprocessing": plan.preprocessing,
            "algorithms": sorted(plan.algorithms),
            "tuning": plan.tuning,
        }, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------- run
    def run(self, dataset: Any, target: str) -> AgentReport:
        state = AgentState()
        self._emit("started", state, "Autonomous Model Doctor started")

        # Profile once. If this fails, the system cannot safely continue.
        try:
            state.profile = self.modules.analyzer.profile(dataset, target)
        except Exception as exc:
            logger.exception("Dataset profiling failed")
            state.errors.append(f"Dataset profiling failed: {exc}")
            state.stop_reason = "Dataset profiling failed"
            self._emit("failed", state, state.stop_reason, {"error": str(exc)})
            return self._report(state)

        self._emit("profiled", state, "Dataset profiling completed", {
            "n_rows": state.profile.n_rows,
            "n_cols": state.profile.n_cols,
            "issues": [asdict(i) for i in state.profile.issues],
        })

        while True:
            stop_reason = self.stop_policy.should_stop(state)
            if stop_reason:
                state.stop_reason = stop_reason
                break

            state.iteration += 1
            try:
                stop_reason = self._run_iteration(state, dataset, target)
            except Exception as exc:
                state.consecutive_errors += 1
                error_message = f"Iteration {state.iteration} failed: {exc}"
                state.errors.append(error_message)
                logger.exception(error_message)
                self._emit("iteration_error", state, error_message)
                continue

            if stop_reason:
                state.stop_reason = stop_reason
                break

        if not state.stop_reason:
            state.stop_reason = "Stopped"
        report = self._report(state)
        self._emit("stopped" if report.status == "completed" else "failed",
                   state, state.stop_reason, {"status": report.status})
        return report

    # -------------------------------------------------------------- iteration
    def _run_iteration(self, state: AgentState, dataset: Any, target: str) -> Optional[str]:
        """Run one iteration. Returns a stop reason if the run must end, else None."""
        m = self.modules
        history = m.memory.history()

        state.last_diagnosis = m.diagnoser.diagnose(state.profile, history)
        self._emit("diagnosed", state, "Model diagnosis completed", {
            "root_cause": state.last_diagnosis.root_cause,
            "explanation": state.last_diagnosis.explanation,
            "problems": [asdict(i) for i in state.last_diagnosis.problems],
        })

        def next_plan(hist: list) -> ActionPlan:
            plan = m.planner.plan(state.profile, state.last_diagnosis, hist)
            if not plan.stop and not plan.algorithms:
                raise ValueError("Plan has no algorithms and does not request stop")
            return plan

        plan = next_plan(history)

        # Ask for an alternative while the plan was already tried (planner stop wins).
        attempts = 0
        while (not plan.stop and m.memory.has_tried(self._plan_signature(plan))
               and attempts < MAX_REPLAN_ATTEMPTS):
            attempts += 1
            self._emit("replanning", state, f"Plan already tried; requesting alternative ({attempts})")
            plan = next_plan(m.memory.history())

        if plan.stop:
            self._safe_log_decision(state, plan)
            return plan.stop_reason or "Planner requested stop"

        signature = self._plan_signature(plan)
        if m.memory.has_tried(signature):
            state.duplicate_plans += 1
            self._emit("duplicate_plan", state, "Plan was already tried; skipping this iteration")
            return None

        self._safe_log_decision(state, plan)
        self._emit("planned", state, plan.reasoning or "Action plan created", {
            "algorithms": plan.algorithms,
            "preprocessing": plan.preprocessing,
            "tuning": plan.tuning,
            "reasoning": plan.reasoning,
        })

        prepared = m.preprocessor.run(dataset, target, plan)
        models = m.trainer.train(prepared, plan)
        if not models:
            raise RuntimeError("Trainer returned no models")

        results = m.evaluator.evaluate(models, prepared, plan)
        if not results:
            raise RuntimeError("Evaluator returned no experiment results")

        # Validate every result before mutating any state or memory.
        scores = [self.stop_policy.score(r) for r in results]

        score_before = state.best_score
        for result, score in zip(results, scores):
            result.iteration = state.iteration
            m.memory.log_experiment(result, plan)
            is_best = self.stop_policy.update_best(state, result)
            if is_best:
                m.memory.save_best(result)
            self._emit("experiment_completed", state, f"Evaluated {result.algorithm}", {
                "algorithm": result.algorithm,
                "metric": self.stop_policy.primary_metric,
                "score": score,
                "metrics": dict(result.metrics),
                "train_metrics": dict(result.train_metrics),
                "is_best": is_best,
            })

        # Mark as tried only after a successful run, so failed plans can be retried.
        m.memory.mark_tried(signature)
        state.consecutive_errors = 0
        progressed = self.stop_policy.finish_iteration(state, score_before)
        self._emit("iteration_completed", state, "Iteration completed", {
            "best_score": state.best_score,
            "progressed": progressed,
            "iterations_without_improvement": state.iterations_without_improvement,
        })
        return None

    def _safe_log_decision(self, state: AgentState, plan: ActionPlan) -> None:
        state.decisions.append({
            "iteration": state.iteration,
            "root_cause": state.last_diagnosis.root_cause if state.last_diagnosis else None,
            "algorithms": list(plan.algorithms),
            "preprocessing": plan.preprocessing,
            "tuning": plan.tuning,
            "reasoning": plan.reasoning,
            "stop": plan.stop,
            "stop_reason": plan.stop_reason,
        })
        try:
            self.modules.memory.log_decision(state.iteration, plan, state.last_diagnosis)
        except Exception:
            logger.exception("Could not persist decision; continuing run")

    def _report(self, state: AgentState) -> AgentReport:
        try:
            history = self.modules.memory.history()
        except Exception:
            history = []
        return AgentReport(
            status="completed" if state.best_result is not None else "failed",
            stop_reason=state.stop_reason,
            iterations=state.iteration,
            best_result=state.best_result,
            profile=state.profile,
            last_diagnosis=state.last_diagnosis,
            errors=state.errors,
            history=history,
            elapsed_seconds=state.elapsed_seconds,
            decisions=state.decisions,
        )
