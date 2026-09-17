# ============================================================
# NAARIRA AGENTIC AI BACKEND
# Phase D1 — Structured Chat API
# ============================================================

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.naarira_agent import (
    get_naarira_agent,
    run_naarira_agent,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("naarira")


# ============================================================
# APP LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("=" * 70)
    logger.info("STARTING NAARIRA AI BACKEND")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Initialize agent once during application startup.
    # --------------------------------------------------------

    try:
        await get_naarira_agent()

        logger.info("✓ Naarira agent initialized successfully")

    except Exception as exc:

        # IMPORTANT:
        # Do not prevent FastAPI from starting just because
        # Gemini/MCP is temporarily unavailable.
        #
        # The request endpoint will surface the actual error.

        logger.exception(
            "Agent initialization failed: %s",
            exc,
        )

    logger.info("=" * 70)
    logger.info("NAARIRA BACKEND READY")
    logger.info("=" * 70)

    yield

    logger.info("Shutting down Naarira backend...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Naarira Agentic AI",
    description=(
        "Agentic AI backend for Naarira using "
        "RAG, MCP, Gemini and PostgreSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer message",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional frontend session identifier",
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "success": True,
        "service": "Naarira Agentic AI",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy",
        "service": "naarira-agentic-ai",
    }


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        print("\n" + "=" * 60)
        print("CHAT REQUEST")
        print("message:", request.message)
        print("session_id:", request.session_id)
        print("=" * 60)

        result = await run_naarira_agent(
            user_message=request.message,
            session_id=request.session_id,
        )

        print("CHAT RESULT:", result)

        return result

    except Exception as e:
        import traceback

        print("\n❌ CHAT ERROR")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))
        traceback.print_exc()

        return {
            "success": False,
            "type": "error",
            "answer": "Sorry, something went wrong while processing your request.",
            "products": [],
            "order": None,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e),
                "error_type": type(e).__name__,
            },
        }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )