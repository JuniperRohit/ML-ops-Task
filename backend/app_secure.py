"""Backend API with authentication."""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from pydantic import BaseModel

from backend.rag import retrieve_context
from backend.rohit_agent import CrewAIOrchestrator, create_rohit_react_agent
from agentic_mlops.auth import verify_token, create_user_token

app = FastAPI(title="ROHIT AI MLOps Assistant")

session_memory = {}
crew_orchestrator = CrewAIOrchestrator()
security = HTTPBearer()


# ===== Authentication =====

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str  # In production, use proper password hashing


class AskRequest(BaseModel):
    """Request model for ask endpoint."""
    question: str
    session_id: str = "default"


class AskResponse(BaseModel):
    """Response model for ask endpoint."""
    question: str
    answer: str
    context: str
    session_memory: dict


class CrewAskResponse(BaseModel):
    """Response model for crew-ask endpoint."""
    question: str
    analyst: str
    explainer: str
    context: str
    session_memory: dict


# ===== Helper Functions =====

async def get_current_user(credentials: Optional[HTTPAuthCredentials] = Depends(security)):
    """
    Get current authenticated user from token.
    
    Leave credentials optional to allow both authenticated and public access.
    """
    if credentials is None:
        return None  # Public access
    
    token = credentials.credentials
    user = verify_token(token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


# ===== Endpoints =====

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/login")
def login(request: LoginRequest):
    """
    Login endpoint - returns JWT token.
    
    In production, verify username/password against secure database.
    """
    # Simple validation (in production, use proper authentication)
    if not request.username or len(request.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username"
        )
    
    # Create token
    token = create_user_token(request.username, email=f"{request.username}@example.com")
    return token


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    current_user=Depends(get_current_user)  # Optional auth check
):
    """
    Ask endpoint with optional authentication.
    
    If auth is provided, token must be valid.
    If no auth, allows public access.
    """
    context = retrieve_context(request.question, top_k=3)
    agent = create_rohit_react_agent()
    try:
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
async def crew_ask(
    request: AskRequest,
    current_user=Depends(get_current_user)  # Optional auth check
):
    """
    Crew ask endpoint with multi-agent orchestration.
    
    If auth is provided, token must be valid.
    If no auth, allows public access.
    """
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


@app.get("/metrics")
async def get_metrics(current_user=Depends(get_current_user)):
    """
    Get system metrics and monitoring data.
    
    Requires authentication.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for metrics"
        )
    
    return {
        "active_sessions": len(session_memory),
        "total_questions": sum(len(msgs) for msgs in session_memory.values()),
        "user": current_user.sub
    }
