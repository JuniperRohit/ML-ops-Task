import pytest

from agentic_mlops.executor import AgenticExecutor
from agentic_mlops.planner import AgenticPlanner


def test_plan_includes_expected_steps():
    planner = AgenticPlanner()
    plan = planner.infer_plan("Train and deploy a model for fraud detection")
    names = [a.name for a in plan.actions]
    assert "generate_dataset" in names
    assert "train_model" in names
    assert "evaluate_model" in names
    assert "deploy_model" in names


def test_execute_flow_success():
    planner = AgenticPlanner()
    executor = AgenticExecutor()

    plan = planner.infer_plan("Train and evaluate model")
    results = executor.execute_plan(plan)
    assert all(r.status != "failed" for r in results)
    assert any(r.name == "evaluate_model" for r in results)
