from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.naarira_agent import (
    create_naarira_agent,
    run_naarira_agent,
)


# ============================================================
# GLOBAL AGENT
# ============================================================

agent = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the Naarira agent once when FastAPI starts.
    """

    global agent

    print("=" * 60)
    print("STARTING NAARIRA AI BACKEND")
    print("=" * 60)

    try:

        print("\nInitializing Naarira Agent...")

        agent = await create_naarira_agent()

        print(
            "\nNaarira Agent initialized successfully."
        )

    except Exception as e:

        print(
            "\nFailed to initialize Naarira Agent:"
        )

        print(
            f"{type(e).__name__}: {str(e)}"
        )

        # Keep FastAPI running so the health endpoint
        # can still report that the agent failed.

        agent = None

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print(
        "\nShutting down Naarira AI backend..."
    )

    agent = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Naarira AI Shopping Agent",
    description=(
        "Agentic AI backend for Naarira using "
        "LangGraph, RAG, MCP and Gemini."
    ),
    version="4.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

# Development configuration.
#
# Later, when deploying:
# allow_origins=["https://naarira.com"]

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
    """
    Request body for /api/chat
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer's message.",
    )


# ============================================================
# PRODUCT RESPONSE MODEL
# ============================================================

class ProductResponse(BaseModel):

    product_id: int | None = None

    title: str

    handle: str | None = None

    category: str | None = None

    url: str


# ============================================================
# CHAT RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    answer: str

    products: list[ProductResponse]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Naarira AI Shopping Agent",
        "status": "running",
        "version": "4.0.0",
        "docs": "/docs",
        "chat_endpoint": "/api/chat",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "agent_initialized": agent is not None,
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):

    global agent

    # --------------------------------------------------------
    # Check agent
    # --------------------------------------------------------

    if agent is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Naarira AI Agent is not initialized."
            ),
        )

    try:

        print(
            "\n" + "=" * 60
        )

        print(
            "CHAT REQUEST"
        )

        print(
            f"Message: {request.message}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Run LangGraph Agent
        # ----------------------------------------------------

        result = await run_naarira_agent(
            query=request.message,
            agent=agent,
            debug=True,
        )

        # ----------------------------------------------------
        # Return structured response
        # ----------------------------------------------------

        response = {
            "answer": result.get(
                "answer",
                "Sorry, I could not generate a response.",
            ),
            "products": result.get(
                "products",
                [],
            ),
        }

        print(
            "\nChat request completed."
        )

        print(
            f"Products returned: "
            f"{len(response['products'])}"
        )

        return response

    except Exception as e:

        print(
            "\n" + "=" * 60
        )

        print(
            "CHAT API ERROR"
        )

        print(
            f"{type(e).__name__}: {str(e)}"
        )

        print(
            "=" * 60
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the chat request."
            ),
        )