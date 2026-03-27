import json

import typer

from .executor import AgenticExecutor
from .planner import AgenticPlanner

app = typer.Typer()
planner = AgenticPlanner()
executor = AgenticExecutor()


@app.command()
def plan(objective: str, context: str = "{}"):
    """Generate an agentic plan for the objective."""
    context_obj = json.loads(context)
    plan = planner.infer_plan(objective, context_obj)
    typer.echo(plan.model_dump_json(indent=2))


@app.command()
def execute(objective: str, context: str = "{}"):
    """Run the plan end-to-end."""
    context_obj = json.loads(context)
    plan = planner.infer_plan(objective, context_obj)
    results = executor.execute_plan(plan, context_obj)
    typer.echo("Plan executed")
    typer.echo(json.dumps([r.model_dump() for r in results], indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8030):
    """Start API server."""
    import uvicorn

    uvicorn.run("agentic_mlops.api:app", host=host, port=port, log_level="info")


def main():
    app()
