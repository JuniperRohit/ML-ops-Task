from typing import Dict, List

from .schemas import Plan, TaskResult, TaskStatus
from .skills import SkillRegistry


class AgenticExecutor:
    def execute_plan(self, plan: Plan, config: Dict[str, any] = None) -> List[TaskResult]:
        config = config or {}
        results: List[TaskResult] = []

        for action in plan.actions:
            task_name = action.name
            skill = SkillRegistry.get(task_name)
            if not skill:
                results.append(
                    TaskResult(
                        name=task_name,
                        task_type=action.task_type,
                        status=TaskStatus.failed,
                        details={"error": "skill_not_found"},
                    )
                )
                continue

            try:
                result = skill(plan.objective, {**config, **action.parameters})
                results.append(result)
            except Exception as exc:
                results.append(
                    TaskResult(
                        name=task_name,
                        task_type=action.task_type,
                        status=TaskStatus.failed,
                        details={"error": str(exc)},
                    )
                )
                break

            if result.status != TaskStatus.success:
                # Early stop on failure for safety and traceability
                break

        return results
