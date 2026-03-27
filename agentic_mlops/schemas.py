from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TaskType(str, Enum):
    data = "data"
    training = "training"
    evaluation = "evaluation"
    deployment = "deployment"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


class TaskAction(BaseModel):
    name: str
    task_type: TaskType
    parameters: Dict[str, Any] = {}


class Plan(BaseModel):
    objective: str
    actions: List[TaskAction] = []


class TaskResult(BaseModel):
    name: str
    task_type: TaskType
    status: TaskStatus
    details: Optional[Dict[str, Any]] = None
