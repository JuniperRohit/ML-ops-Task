import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag import retrieve_context
from .rohit_agent import CrewAIOrchestrator, create_rohit_react_agent

app = FastAPI(title="ROHIT AI MLOps Assistant")

session_memory = {}
crew_orchestrator = CrewAIOrchestrator()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class AskResponse(BaseModel):
    question: str
    answer: str
    context: str
    session_memory: dict


class CrewAskResponse(BaseModel):
    question: str
    analyst: str
    explainer: str
    context: str
    session_memory: dict


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    context = retrieve_context(request.question, top_k=3)
    agent = create_rohit_react_agent()
    try:
        # Use invoke instead of run
        response = agent.invoke(
            {"input": f"Answer the question with MLOps context. If needed, use the KnowledgeBase tool.\nQuestion: {request.question}"}
        )
        answer = response.output if hasattr(response, 'output') else str(response)
    except Exception as exc:
        logging.exception("Ask endpoint agent error")
        answer = f"Error: {str(exc)}"

    session_memory.setdefault(request.session_id, []).append({"question": request.question, "answer": answer})
    return AskResponse(
        question=request.question,
        answer=answer,
        context=context,
        session_memory={"session_id": request.session_id, "messages": session_memory[request.session_id]},
    )


@app.post("/crew-ask", response_model=CrewAskResponse)
async def crew_ask(request: AskRequest):
    context = retrieve_context(request.question, top_k=3)
    try:
        crew = crew_orchestrator.run(request.question, session_memory.setdefault(request.session_id, {}))
    except Exception as exc:
        logging.exception("Crew Ask endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))

    return CrewAskResponse(
        question=request.question,
        analyst=crew["analyst"],
        explainer=crew["explainer"],
        context=context,
        session_memory=session_memory[request.session_id],
    )
