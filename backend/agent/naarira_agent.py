# backend/agent/naarira_agent.py

import os
import json
import traceback
import re
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from rag.search import semantic_search


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing in environment variables."
    )


# MCP URL
#
# Production:
# MCP_URL=https://your-mcp-service.onrender.com/mcp
#
# Local:
# MCP_URL=http://127.0.0.1:8001/mcp
#
MCP_URL = os.getenv(
    "MCP_URL",
    "http://127.0.0.1:8001/mcp",
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Naarira AI, the shopping assistant for Naarira,
a women's ethnic fashion store.

Naarira sells products such as:

- Sarees
- Suits
- Dresses
- Kurtis
- Lehengas
- Ethnic wear

Your job is to help customers discover products and answer
product-related questions using ONLY reliable catalog data.

IMPORTANT RULES
================

1. NEVER invent products.
2. NEVER invent prices.
3. NEVER invent sizes.
4. NEVER invent colors.
5. NEVER invent inventory.
6. NEVER invent product URLs.
7. NEVER claim a product is available unless the catalog/tool
   data confirms it.
8. Use MCP for structured catalog information.
9. Use RAG for semantic product discovery.
10. For exact structured filters such as:
    - price
    - color
    - size
    - category
    - availability

    prefer the MCP search tool.

11. For natural language / semantic requests, use RAG.
12. For combined requests, use both MCP and RAG when useful.

Examples:

"Show designer sarees"
    → RAG can help discover semantically relevant products.

"Show sarees under 2000"
    → MCP structured filtering.

"Show black sarees under 2000"
    → MCP structured filtering.

"Show elegant designer sarees under 2000"
    → MCP for price restriction,
      then RAG for semantic ranking.

"Is this saree available in red?"
    → MCP product details / availability.

13. When presenting products, return product information
    separately from the natural-language answer.

14. Product URLs must be real URLs returned by the tools.

15. Never create Markdown links containing invented URLs.

16. Keep responses useful and concise.

17. If no matching products are found, clearly say that no
    matching products were found.

18. Do not expose internal implementation details unless
    explicitly asked.

19. You can use the available MCP tools:
    - search_products_tool
    - get_product_details_tool
    - check_product_availability_tool

20. You can use the RAG tool:
    - search_naarira_products_with_rag

21. For availability questions, use the availability tool
    instead of guessing from semantic search.

22. For exact product details, use MCP rather than relying
    only on RAG.
"""


# ============================================================
# RAG TOOL
# ============================================================

async def search_naarira_products_with_rag(
    query: str,
    candidate_product_ids: list[int] | None = None,
) -> list[dict]:
    """
    Semantic product search using Gemini embeddings
    and PostgreSQL + pgvector.

    candidate_product_ids can optionally restrict the semantic
    search to a set of product IDs returned by MCP.
    """

    try:
        print()
        print("=" * 60)
        print("RAG SEARCH")
        print("=" * 60)

        print(f"Query: {query}")
        print(
            f"Candidate IDs: {candidate_product_ids}"
        )

        results = semantic_search(
            query=query,
            top_k=5,
            candidate_product_ids=candidate_product_ids,
        )

        print(
            f"RAG results: {len(results)}"
        )

        return results

    except Exception as exc:

        print("=" * 60)
        print("RAG SEARCH FAILED")
        print("=" * 60)

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        return []


# ============================================================
# MCP CLIENT
# ============================================================

async def create_mcp_client():
    """
    Create the MCP client configuration.

    The MCP server is exposed through Streamable HTTP.

    Production example:

        MCP_URL=https://your-mcp-service.onrender.com/mcp

    Local example:

        MCP_URL=http://127.0.0.1:8001/mcp
    """

    print()
    print("=" * 60)
    print("CONNECTING TO NAARIRA MCP SERVER")
    print("=" * 60)

    print(
        f"MCP URL: {MCP_URL}"
    )

    try:

        client = MultiServerMCPClient(
            {
                "naarira": {
                    "transport": "http",
                    "url": MCP_URL,
                }
            }
        )

        print(
            "MCP client created successfully."
        )

        return client

    except Exception as exc:

        print("=" * 60)
        print("MCP CLIENT CREATION FAILED")
        print("=" * 60)

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        raise


# ============================================================
# CREATE AGENT
# ============================================================

async def create_naarira_agent():
    """
    Initialize the complete Naarira agent.

    Components:

        Gemini LLM
        +
        RAG tool
        +
        MCP tools
        +
        LangGraph agent
    """

    print()
    print("=" * 60)
    print("INITIALIZING NAARIRA AGENT")
    print("=" * 60)

    print(
        f"Gemini model: {MODEL_NAME}"
    )

    print(
        f"MCP URL: {MCP_URL}"
    )

    try:

        # ----------------------------------------------------
        # Create MCP client
        # ----------------------------------------------------

        mcp_client = await create_mcp_client()

        # ----------------------------------------------------
        # Load MCP tools
        # ----------------------------------------------------

        print()
        print(
            "Loading MCP tools..."
        )

        mcp_tools = await mcp_client.get_tools()

        print(
            f"Loaded {len(mcp_tools)} MCP tools."
        )

        for tool in mcp_tools:

            print(
                f"  - {tool.name}"
            )

        # ----------------------------------------------------
        # Combine tools
        # ----------------------------------------------------

        all_tools = [
            search_naarira_products_with_rag,
            *mcp_tools,
        ]

        print()
        print(
            "Total agent tools:",
            len(all_tools),
        )

        for tool in all_tools:
            tool_name = getattr(tool, "name", None)

    if not tool_name:
        tool_name = getattr(tool, "__name__", str(tool))

    print(f"  - {tool_name}")

        # ----------------------------------------------------
        # Create LangGraph agent
        # ----------------------------------------------------

        print()
        print(
            "Creating LangGraph agent..."
        )

        agent = create_agent(
            model=model,
            tools=all_tools,
            system_prompt=SYSTEM_PROMPT,
            name="naarira_shopping_agent",
        )

        print()
        print("=" * 60)
        print("NAARIRA AGENT INITIALIZED SUCCESSFULLY")
        print("=" * 60)

        return agent

    except Exception as exc:

        print()
        print("=" * 60)
        print("FAILED TO INITIALIZE NAARIRA AGENT")
        print("=" * 60)

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        print()
        print("FULL TRACEBACK:")
        print("-" * 60)

        traceback.print_exc()

        print("-" * 60)

        raise


# ============================================================
# JSON EXTRACTION HELPERS
# ============================================================

def _extract_json_objects(text: str) -> list[dict]:
    """
    Extract JSON objects from model output.

    Handles:
        plain JSON
        ```json ... ```
        JSON embedded in surrounding text
    """

    if not text:
        return []

    text = text.strip()

    results = []

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            results.append(parsed)

        elif isinstance(parsed, list):

            results.extend(
                item
                for item in parsed
                if isinstance(item, dict)
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Markdown JSON block
    # --------------------------------------------------------

    code_blocks = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for block in code_blocks:

        try:

            parsed = json.loads(
                block.strip()
            )

            if isinstance(parsed, dict):

                results.append(parsed)

            elif isinstance(parsed, list):

                results.extend(
                    item
                    for item in parsed
                    if isinstance(item, dict)
                )

        except Exception:

            continue

    # --------------------------------------------------------
    # Generic {...} extraction
    # --------------------------------------------------------

    for match in re.finditer(
        r"\{.*?\}",
        text,
        flags=re.DOTALL,
    ):

        candidate = match.group(0)

        try:

            parsed = json.loads(candidate)

            if isinstance(parsed, dict):

                results.append(parsed)

        except Exception:

            continue

    return results


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_response(result: Any) -> str:
    """
    Extract readable text from LangGraph agent result.
    """

    if result is None:
        return ""

    # --------------------------------------------------------
    # Dictionary result
    # --------------------------------------------------------

    if isinstance(result, dict):

        # Common LangGraph format
        messages = result.get(
            "messages"
        )

        if messages:

            return extract_response(
                {
                    "messages": messages
                }
            )

        # Direct answer
        for key in (
            "answer",
            "output",
            "response",
            "content",
        ):

            value = result.get(key)

            if isinstance(value, str):
                return value

        return str(result)

    # --------------------------------------------------------
    # Message list
    # --------------------------------------------------------

    if isinstance(result, list):

        # Search backwards for final assistant message
        for message in reversed(result):

            if isinstance(message, dict):

                content = message.get(
                    "content"
                )

                if isinstance(content, str):

                    return content

                if isinstance(content, list):

                    parts = []

                    for item in content:

                        if isinstance(item, dict):

                            text = item.get(
                                "text"
                            )

                            if text:
                                parts.append(
                                    text
                                )

                    if parts:

                        return "\n".join(parts)

            else:

                content = getattr(
                    message,
                    "content",
                    None,
                )

                if isinstance(content, str):

                    return content

        return ""

    # --------------------------------------------------------
    # Object with content
    # --------------------------------------------------------

    content = getattr(
        result,
        "content",
        None,
    )

    if isinstance(content, str):
        return content

    return str(result)


# ============================================================
# PRODUCT EXTRACTION
# ============================================================

def extract_products_from_result(
    result: Any,
) -> list[dict]:
    """
    Extract structured product information from
    agent result / tool messages.

    The function intentionally accepts multiple
    formats because MCP and LangGraph can return
    slightly different message structures.
    """

    products = []

    # --------------------------------------------------------
    # Recursive collector
    # --------------------------------------------------------

    def collect(value: Any):

        if value is None:
            return

        # -----------------------------------------------
        # Dictionary
        # -----------------------------------------------

        if isinstance(value, dict):

            # Direct product
            if (
                "product_id" in value
                or "id" in value
            ) and (
                "title" in value
                or "name" in value
            ):

                product = {
                    "product_id": (
                        value.get("product_id")
                        or value.get("id")
                    ),
                    "title": (
                        value.get("title")
                        or value.get("name")
                    ),
                    "handle": value.get(
                        "handle"
                    ),
                    "category": value.get(
                        "category"
                    ),
                    "url": value.get(
                        "url"
                    )
                    or value.get(
                        "product_url"
                    ),
                }

                # Keep product only when title exists
                if product["title"]:

                    products.append(
                        product
                    )

            # Search common product containers
            for key in (
                "products",
                "results",
                "items",
                "data",
                "matches",
            ):

                if key in value:

                    collect(
                        value[key]
                    )

            # Recurse through all values where useful
            for key, item in value.items():

                if key in {
                    "products",
                    "results",
                    "items",
                    "data",
                    "matches",
                }:
                    continue

                if isinstance(
                    item,
                    (dict, list),
                ):

                    collect(item)

            return

        # -----------------------------------------------
        # List
        # -----------------------------------------------

        if isinstance(value, list):

            for item in value:

                collect(item)

            return

        # -----------------------------------------------
        # String containing JSON
        # -----------------------------------------------

        if isinstance(value, str):

            parsed_objects = (
                _extract_json_objects(
                    value
                )
            )

            for obj in parsed_objects:

                collect(obj)

            return

        # -----------------------------------------------
        # Object with content
        # -----------------------------------------------

        content = getattr(
            value,
            "content",
            None,
        )

        if content is not None:

            collect(content)

    # Start collection
    collect(result)

    # --------------------------------------------------------
    # De-duplicate
    # --------------------------------------------------------

    unique_products = []

    seen = set()

    for product in products:

        product_id = product.get(
            "product_id"
        )

        key = (
            str(product_id)
            if product_id is not None
            else (
                product.get("title")
                or ""
            ).lower()
        )

        if key in seen:
            continue

        seen.add(key)

        unique_products.append(
            product
        )

    return unique_products


# ============================================================
# CLEAN PRODUCT OUTPUT
# ============================================================

def normalize_products(
    products: list[dict],
) -> list[dict]:
    """
    Normalize product output for FastAPI/frontend.
    """

    cleaned = []

    for product in products:

        title = product.get(
            "title"
        )

        if not title:
            continue

        product_id = product.get(
            "product_id"
        )

        try:

            if product_id is not None:

                product_id = int(
                    product_id
                )

        except Exception:

            product_id = None

        handle = product.get(
            "handle"
        )

        category = product.get(
            "category"
        )

        url = product.get(
            "url"
        ) or product.get(
            "product_url"
        )

        # If an actual URL was not returned,
        # don't invent one.
        if not url:

            if handle:

                url = (
                    "https://naarira.com/products/"
                    f"{handle}"
                )

        if not url:

            continue

        cleaned.append(
            {
                "product_id": product_id,
                "title": str(title),
                "handle": handle,
                "category": category,
                "url": str(url),
            }
        )

    return cleaned


# ============================================================
# AGENT RUNNER
# ============================================================

async def run_naarira_agent(
    query: str,
    agent=None,
    debug: bool = False,
) -> dict:
    """
    Run a user query through the Naarira Agent.

    Returns:

    {
        "answer": "...",
        "products": [...]
    }
    """

    if not query or not query.strip():

        return {
            "answer": (
                "Please tell me what you're looking for."
            ),
            "products": [],
        }

    query = query.strip()

    print()
    print("=" * 60)
    print("RUNNING NAARIRA AGENT")
    print("=" * 60)

    print(
        f"User query: {query}"
    )

    try:

        # ----------------------------------------------------
        # Create agent when not supplied
        # ----------------------------------------------------

        if agent is None:

            agent = await create_naarira_agent()

        # ----------------------------------------------------
        # Invoke LangGraph
        # ----------------------------------------------------

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )

        # ----------------------------------------------------
        # Extract final answer
        # ----------------------------------------------------

        answer = extract_response(
            result
        )

        # ----------------------------------------------------
        # Extract products
        # ----------------------------------------------------

        products = (
            extract_products_from_result(
                result
            )
        )

        products = normalize_products(
            products
        )

        # ----------------------------------------------------
        # Debug output
        # ----------------------------------------------------

        if debug:

            print()
            print("=" * 60)
            print("AGENT ANSWER")
            print("=" * 60)

            print(answer)

            print()
            print(
                "PRODUCTS:"
            )

            print(
                json.dumps(
                    products,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        return {
            "answer": answer,
            "products": products,
        }

    except Exception as exc:

        print()
        print("=" * 60)
        print("NAARIRA AGENT EXECUTION FAILED")
        print("=" * 60)

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        return {
            "answer": (
                "I'm sorry, but I couldn't process "
                "your request right now. Please try again."
            ),
            "products": [],
        }


# ============================================================
# AGENT TRACE
# ============================================================

def print_agent_trace(
    result: Any,
):
    """
    Print LangGraph messages for debugging.
    """

    print()
    print("=" * 60)
    print("AGENT TRACE")
    print("=" * 60)

    if not isinstance(
        result,
        dict,
    ):

        print(result)
        return

    messages = result.get(
        "messages",
        [],
    )

    for index, message in enumerate(
        messages,
        start=1,
    ):

        print()
        print(
            f"--- MESSAGE {index} ---"
        )

        # LangChain message
        message_type = getattr(
            message,
            "type",
            None,
        )

        if message_type:

            print(
                f"type: {message_type}"
            )

        name = getattr(
            message,
            "name",
            None,
        )

        if name:

            print(
                f"name: {name}"
            )

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if tool_calls:

            print(
                "tool_calls:"
            )

            print(
                json.dumps(
                    tool_calls,
                    indent=2,
                    default=str,
                )
            )

        content = getattr(
            message,
            "content",
            None,
        )

        if content:

            print(
                "content:"
            )

            if isinstance(
                content,
                str,
            ):

                print(content)

            else:

                print(
                    json.dumps(
                        content,
                        indent=2,
                        default=str,
                    )
                )


# ============================================================
# CLI TEST
# ============================================================

async def main():
    """
    Local command-line test.
    """

    print()
    print("=" * 70)
    print("NAARIRA AGENT TEST")
    print("=" * 70)

    print(
        f"MCP URL: {MCP_URL}"
    )

    print(
        f"Gemini model: {MODEL_NAME}"
    )

    print()

    query = input(
        "Ask Naarira AI: "
    ).strip()

    if not query:

        print(
            "No query provided."
        )

        return

    try:

        # Create agent
        agent = await create_naarira_agent()

        # Run query
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )

        # Print trace
        print_agent_trace(
            result
        )

        # Print clean response
        answer = extract_response(
            result
        )

        products = extract_products_from_result(
            result
        )

        products = normalize_products(
            products
        )

        print()
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

        print()
        print("=" * 70)
        print("PRODUCTS")
        print("=" * 70)

        print(
            json.dumps(
                products,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import asyncio

    asyncio.run(
        main()
    )