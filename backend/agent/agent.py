import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


class AgentState(TypedDict):
    message: str
    response: str


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


def call_gemini(state: AgentState) -> AgentState:

    response = llm.invoke(
        state["message"]
    )

    return {
        **state,
        "response": response.content
    }


graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "gemini",
    call_gemini
)

graph_builder.add_edge(
    START,
    "gemini"
)

graph_builder.add_edge(
    "gemini",
    END
)

agent = graph_builder.compile()


if __name__ == "__main__":

    print("=" * 60)
    print("NAARIRA LANGGRAPH AGENT TEST")
    print("=" * 60)

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question.")
    else:

        result = agent.invoke({
            "message": query,
            "response": ""
        })

        print("\nAgent Response:")
        print(result["response"])

        print("\n" + "=" * 60)
        print("AGENT TEST COMPLETE")
        print("=" * 60)