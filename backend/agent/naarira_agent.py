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


# Production:
# MCP_URL=https://your-mcp-service.onrender.com/mcp
#
# Local:
# MCP_URL=http://127.0.0.1:8001/mcp

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

6. NEVER invent product availability.

7. NEVER invent product URLs.

8. Use MCP for structured catalog information such as:

   - price
   - color
   - size
   - category
   - stock
   - availability

9. Use RAG for semantic product discovery.

10. For combined queries, use MCP for hard constraints and
    RAG for semantic relevance when useful.

Examples:

"Show designer sarees"
    -> RAG can help discover semantically relevant products.

"Show sarees under 2000"
    -> MCP structured filtering.

"Show black sarees under 2000"
    -> MCP structured filtering.

"Show elegant designer sarees under 2000"
    -> MCP for price restriction,
       then RAG for semantic ranking.

"Is this saree available in red?"
    -> MCP product details / availability.

11. For availability questions, use the availability tool.

12. For exact product details, use MCP product details.

13. VERY IMPORTANT:
    DO NOT put product URLs inside the natural-language answer.

14. DO NOT generate Markdown links.

15. DO NOT generate raw URLs.

16. DO NOT write:

    [View Product](https://...)
    [Link](https://...)
    https://naarira.com/...

17. Product URLs must be returned separately as structured
    product data.

18. The frontend will display products as product cards with
    a "View Product" button.

19. The natural-language answer should remain clean,
    conversational and concise.

20. Do not manually create a numbered product list containing
    URLs.

21. If products are found, let the structured product results
    contain their product data.

22. If no matching products are found, clearly say that no
    matching products were found.

23. Do not expose internal Agent, RAG, MCP or database details
    unless explicitly asked.
"""


# ============================================================
# RAG TOOL
# ============================================================

async def search_naarira_products_with_rag(
    query: str,
    candidate_product_ids: list[int] | None = None,
) -> list[dict]:
    """
    Semantic product search using Gemini embeddings and
    PostgreSQL + pgvector.

    candidate_product_ids can optionally restrict semantic
    search to a set of MCP-selected products.
    """

    try:

        print()
        print("=" * 60)
        print("RAG SEARCH")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

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

        print()
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

    Production:
        MCP_URL=https://your-mcp-service.onrender.com/mcp

    Local:
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

        print()
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
    Initialize:

    Gemini
       +
    RAG
       +
    MCP
       +
    LangGraph Agent
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
            f"Total agent tools: {len(all_tools)}"
        )

        for tool in all_tools:

            tool_name = getattr(
                tool,
                "name",
                None,
            )

            if not tool_name:

                tool_name = getattr(
                    tool,
                    "__name__",
                    str(tool),
                )

            print(
                f"  - {tool_name}"
            )

        # ----------------------------------------------------
        # Create LangGraph Agent
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

def _extract_json_objects(
    text: str,
) -> list[dict]:

    if not text:
        return []

    text = text.strip()

    results = []

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(
            parsed,
            dict,
        ):

            results.append(
                parsed
            )

        elif isinstance(
            parsed,
            list,
        ):

            results.extend(
                item
                for item in parsed
                if isinstance(
                    item,
                    dict,
                )
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

            if isinstance(
                parsed,
                dict,
            ):

                results.append(
                    parsed
                )

            elif isinstance(
                parsed,
                list,
            ):

                results.extend(
                    item
                    for item in parsed
                    if isinstance(
                        item,
                        dict,
                    )
                )

        except Exception:

            continue

    # --------------------------------------------------------
    # Generic JSON objects
    # --------------------------------------------------------

    for match in re.finditer(
        r"\{.*?\}",
        text,
        flags=re.DOTALL,
    ):

        candidate = match.group(0)

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                dict,
            ):

                results.append(
                    parsed
                )

        except Exception:

            continue

    return results


# ============================================================
# MESSAGE CONTENT NORMALIZATION
# ============================================================

def normalize_message_content(
    content: Any,
) -> str:

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                if item.strip():

                    parts.append(
                        item.strip()
                    )

            elif isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if (
                    text
                    and str(text).strip()
                ):

                    parts.append(
                        str(text).strip()
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_response(
    result: Any,
) -> str:
    """
    Extract final assistant response without recursion.
    """

    if result is None:
        return ""

    # --------------------------------------------------------
    # LangGraph state dictionary
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        messages = result.get(
            "messages"
        )

        if messages:

            return extract_response_from_messages(
                messages
            )

        for key in (
            "answer",
            "output",
            "response",
            "content",
        ):

            value = result.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):

                return value.strip()

        return ""

    # --------------------------------------------------------
    # List of messages
    # --------------------------------------------------------

    if isinstance(
        result,
        list,
    ):

        return extract_response_from_messages(
            result
        )

    # --------------------------------------------------------
    # Single message object
    # --------------------------------------------------------

    content = getattr(
        result,
        "content",
        None,
    )

    if content is not None:

        return normalize_message_content(
            content
        )

    return str(
        result
    )


def extract_response_from_messages(
    messages: list[Any],
) -> str:
    """
    Extract the last meaningful assistant/model message.
    """

    if not messages:
        return ""

    for message in reversed(
        messages
    ):

        # ----------------------------------------------------
        # Dictionary message
        # ----------------------------------------------------

        if isinstance(
            message,
            dict,
        ):

            role = str(
                message.get("role")
                or message.get("type")
                or ""
            ).lower()

            if role in {
                "tool",
                "function",
            }:

                continue

            content = message.get(
                "content"
            )

            text = normalize_message_content(
                content
            )

            if text:

                return text

            continue

        # ----------------------------------------------------
        # LangChain message
        # ----------------------------------------------------

        message_type = str(
            getattr(
                message,
                "type",
                "",
            )
            or ""
        ).lower()

        if message_type in {
            "tool",
            "function",
        }:

            continue

        content = getattr(
            message,
            "content",
            None,
        )

        text = normalize_message_content(
            content
        )

        if text:

            return text

    return ""


# ============================================================
# PRODUCT EXTRACTION
# ============================================================

def extract_products_from_result(
    result: Any,
) -> list[dict]:
    """
    Extract structured products from Agent/MCP/RAG results.
    """

    products = []

    def collect(
        value: Any,
    ):

        if value is None:
            return

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            # Direct product object
            if (
                "product_id" in value
                or "id" in value
            ) and (
                "title" in value
                or "name" in value
            ):

                product = {
                    "product_id": (
                        value.get(
                            "product_id"
                        )
                        or value.get(
                            "id"
                        )
                    ),
                    "title": (
                        value.get(
                            "title"
                        )
                        or value.get(
                            "name"
                        )
                    ),
                    "handle": value.get(
                        "handle"
                    ),
                    "category": value.get(
                        "category"
                    ),
                    "url": (
                        value.get(
                            "url"
                        )
                        or value.get(
                            "product_url"
                        )
                    ),
                    "price": value.get(
                        "price"
                    ),
                    "image_url": (
                        value.get(
                            "image_url"
                        )
                        or value.get(
                            "image"
                        )
                    ),
                    "available": value.get(
                        "available"
                    ),
                }

                if product[
                    "title"
                ]:

                    products.append(
                        product
                    )

            # ------------------------------------------------
            # Common product containers
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Recurse nested objects
            # ------------------------------------------------

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
                    (
                        dict,
                        list,
                    ),
                ):

                    collect(
                        item
                    )

            return

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            for item in value:

                collect(
                    item
                )

            return

        # ----------------------------------------------------
        # String containing JSON
        # ----------------------------------------------------

        if isinstance(
            value,
            str,
        ):

            parsed_objects = (
                _extract_json_objects(
                    value
                )
            )

            for obj in parsed_objects:

                collect(
                    obj
                )

            return

        # ----------------------------------------------------
        # Object containing content
        # ----------------------------------------------------

        content = getattr(
            value,
            "content",
            None,
        )

        if content is not None:

            collect(
                content
            )

    # Start
    collect(
        result
    )

    # --------------------------------------------------------
    # De-duplicate
    # --------------------------------------------------------

    unique_products = []

    seen = set()

    for product in products:

        product_id = product.get(
            "product_id"
        )

        title = product.get(
            "title"
        )

        key = (
            str(product_id)
            if product_id is not None
            else str(
                title or ""
            ).lower()
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_products.append(
            product
        )

    return unique_products


# ============================================================
# NORMALIZE PRODUCTS
# ============================================================

def normalize_products(
    products: list[dict],
) -> list[dict]:
    """
    Normalize structured product response for FastAPI/frontend.
    """

    cleaned = []

    for product in products:

        title = product.get(
            "title"
        )

        if not title:
            continue

        # ----------------------------------------------------
        # Product ID
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Handle
        # ----------------------------------------------------

        handle = product.get(
            "handle"
        )

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        category = product.get(
            "category"
        )

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url = (
            product.get(
                "url"
            )
            or product.get(
                "product_url"
            )
        )

        # Safe Shopify fallback
        if (
            not url
            and handle
        ):

            url = (
                "https://naarira.com/products/"
                f"{handle}"
            )

        # Product must have a usable URL
        if not url:
            continue

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        price = product.get(
            "price"
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_url = (
            product.get(
                "image_url"
            )
            or product.get(
                "image"
            )
        )

        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        available = product.get(
            "available"
        )

        cleaned.append(
            {
                "product_id": product_id,
                "title": str(title),
                "handle": handle,
                "category": category,
                "url": str(url),
                "price": price,
                "image_url": image_url,
                "available": available,
            }
        )

    return cleaned


# ============================================================
# CLEAN AI ANSWER
# ============================================================

def clean_ai_answer(
    answer: str,
) -> str:
    """
    Remove URLs and Markdown product links from the AI answer.

    Product URLs are displayed separately by the frontend.
    """

    if not answer:
        return ""

    clean = str(
        answer
    )

    # --------------------------------------------------------
    # Convert Markdown links to their visible text
    #
    # [View Product](https://...)
    # ->
    # View Product
    # --------------------------------------------------------

    clean = re.sub(
        r"\[([^\]]+)\]\(\s*https?://[^)]+\)",
        r"\1",
        clean,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Remove raw Naarira product URLs
    # --------------------------------------------------------

    clean = re.sub(
        r"https?://(?:www\.)?naarira\.com/products/[^\s<>\])}]+",
        "",
        clean,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Remove explicit "Link:" lines
    # --------------------------------------------------------

    clean = re.sub(
        r"(?im)^\s*(?:link|product link)\s*:\s*.*$",
        "",
        clean,
    )

    # --------------------------------------------------------
    # Clean empty markdown lines
    # --------------------------------------------------------

    clean = re.sub(
        r"\n\s*[-•]\s*\n",
        "\n",
        clean,
    )

    # --------------------------------------------------------
    # Remove excessive blank lines
    # --------------------------------------------------------

    clean = re.sub(
        r"\n{3,}",
        "\n\n",
        clean,
    )

    return clean.strip()


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
        # Create agent if needed
        # ----------------------------------------------------

        if agent is None:

            agent = await create_naarira_agent()

        # ----------------------------------------------------
        # Invoke LangGraph Agent
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
        # Extract answer
        # ----------------------------------------------------

        answer = extract_response(
            result
        )

        # ----------------------------------------------------
        # Clean answer
        # ----------------------------------------------------

        answer = clean_ai_answer(
            answer
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
        # Debug
        # ----------------------------------------------------

        if debug:

            print()
            print("=" * 60)
            print("AGENT ANSWER")
            print("=" * 60)

            print(
                answer
            )

            print()
            print("=" * 60)
            print("PRODUCTS")
            print("=" * 60)

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

    print()
    print("=" * 60)
    print("AGENT TRACE")
    print("=" * 60)

    if not isinstance(
        result,
        dict,
    ):

        print(
            result
        )

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

                print(
                    content
                )

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

        agent = await create_naarira_agent()

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

        print_agent_trace(
            result
        )

        answer = clean_ai_answer(
            extract_response(
                result
            )
        )

        products = (
            extract_products_from_result(
                result
            )
        )

        products = normalize_products(
            products
        )

        print()
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(
            answer
        )

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