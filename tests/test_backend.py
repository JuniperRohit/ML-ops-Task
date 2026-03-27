import os
from fastapi.testclient import TestClient

from backend.app import app, session_memory
from backend.rag import retrieve_context, load_knowledge_from_folder
from backend.rohit_agent import CrewAIOrchestrator, create_rohit_react_agent

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer():
    payload = {"question": "How to use MLflow?", "session_id": "test1"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["question"] == payload["question"]


def test_crew_ask_endpoint_returns_analyst_explainer():
    payload = {"question": "What is DVC?", "session_id": "test2"}
    response = client.post("/crew-ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "analyst" in data
    assert "explainer" in data


def test_session_memory_persists_across_requests():
    sid = "session_test"
    session_memory.clear()
    client.post("/ask", json={"question": "MLflow tracking?", "session_id": sid})
    client.post("/ask", json={"question": "MLflow serve?", "session_id": sid})
    assert len(session_memory[sid]) == 2


def test_retrieve_context_from_knowledge_base():
    count = load_knowledge_from_folder()  # should pull from /knowledge md files
    assert count > 0
    ctx = retrieve_context("MLflow experiment tracking", top_k=2)
    assert isinstance(ctx, str)
    assert "MLflow" in ctx or "mlflow" in ctx.lower()


def test_rohit_react_agent_creates_agent_and_responds():
    agent = create_rohit_react_agent()
    assert agent is not None
    out = agent.run("Find context about MLflow with KnowledgeBase.")
    assert isinstance(out, str)


def test_crewai_orchestrator_flow():
    crew = CrewAIOrchestrator()
    history = {}
    result = crew.run("Explain Kubeflow pipelines.", history)
    assert "analyst" in result
    assert "explainer" in result
    assert history["last_analyst"] == result["analyst"]


def test_ask_endpoint_error_handling_for_nonexisting_session():
    response = client.post("/ask", json={"question": "qa", "session_id": ""})
    assert response.status_code == 200
    assert "answer" in response.json()
