"""Demo: runs the agent with mock modules. Replace the mocks with real modules as they are delivered."""
from agent import Orchestrator, StopPolicy
from agent.mocks import build_mock_modules


def show(event):
    print(f"[iter {event.iteration}] {event.event_type:<20} {event.message}")


orchestrator = Orchestrator(build_mock_modules(), StopPolicy(max_iterations=6, patience=3), on_event=show)
report = orchestrator.run(dataset=None, target="label")

print()
print(f"status={report.status} | iterations={report.iterations} | elapsed={report.elapsed_seconds:.3f}s")
print(f"stop reason: {report.stop_reason}")
if report.best_result:
    b = report.best_result
    print(f"best: {b.algorithm} {b.metrics} (iteration {b.iteration})")
