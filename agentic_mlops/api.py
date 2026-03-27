from fastapi import FastAPI, HTTPException

from .executor import AgenticExecutor
from .planner import AgenticPlanner
from .schemas import Plan, TaskResult

app = FastAPI(title="Agentic MLOps Assistant", version="0.1.0")
planner = AgenticPlanner()
executor = AgenticExecutor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan", response_model=Plan)
def create_plan(objective: str, context: dict = None):
    plan = planner.infer_plan(objective=objective, context=context or {})
    return plan


@app.post("/execute")
def execute_plan(objective: str, context: dict = None):
    plan = planner.infer_plan(objective=objective, context=context or {})
    results = executor.execute_plan(plan=plan, config=context or {})
    return {"plan": plan, "results": results}


@app.post("/run")
def run_full_flow(objective: str, context: dict = None):
    plan = planner.infer_plan(objective=objective, context=context or {})
    results = executor.execute_plan(plan=plan, config=context or {})

    from .schemas import TaskStatus

    if any(r.status != TaskStatus.success for r in results):
        raise HTTPException(status_code=500, detail="One or more steps failed")

    return {"objective": objective, "plan": plan, "results": results}
