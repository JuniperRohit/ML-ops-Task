from typing import Dict, List

from .schemas import Plan, TaskAction, TaskType


class AgenticPlanner:
    """A lightweight, custom plan generator that avoids typical repo patterns."""

    def infer_plan(self, objective: str, context: Dict[str, str] = None) -> Plan:
        context = context or {}
        lower = objective.lower()

        actions: List[TaskAction] = []

        # First steering: ensure data is generated for any prediction demand
        if "train" in lower or "model" in lower or "predict" in lower:
            actions.append(TaskAction(name="generate_dataset", task_type=TaskType.data))
            actions.append(TaskAction(name="train_model", task_type=TaskType.training))
            actions.append(TaskAction(name="evaluate_model", task_type=TaskType.evaluation))

            if "deploy" in lower or "production" in lower or "deliver" in lower:
                actions.append(TaskAction(name="deploy_model", task_type=TaskType.deployment))

        # If no known command, fallback to exploratory baseline pipeline
        if not actions:
            actions = [
                TaskAction(name="generate_dataset", task_type=TaskType.data),
                TaskAction(name="train_model", task_type=TaskType.training),
                TaskAction(name="evaluate_model", task_type=TaskType.evaluation),
            ]

        return Plan(objective=objective, actions=actions)
