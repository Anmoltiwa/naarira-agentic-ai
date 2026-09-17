"""
Naarira Agent

Phase C — Agent Routing
Phase D1 — Structured API Response

Capabilities:
1. Product RAG
2. Policy / FAQ RAG
3. Product MCP tools
4. Order tracking MCP tool

Run from backend:

    python -m agent.naarira_agent
"""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag.policy_rag import generate_policy_answer


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing in environment variables."
    )

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

MCP_URL = os.getenv(
    "MCP_URL",
    "http://127.0.0.1:8001/mcp",
)


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# PRODUCT RAG IMPORT
# ============================================================

PRODUCT_RAG_FUNCTION = None

try:
    import rag.rag as product_rag_module

    possible_names = [
        "run_rag",
        "search_products_with_rag",
        "search_naarira_products_with_rag",
        "rag_search",
        "search_products",
    ]

    for function_name in possible_names:
        candidate = getattr(
            product_rag_module,
            function_name,
            None,
        )

        if callable(candidate):
            PRODUCT_RAG_FUNCTION = candidate

            print(
                f"✓ Product RAG function found: {function_name}"
            )
            break

except Exception as exc:
    print(
        "⚠ Could not import rag.rag:",
        type(exc).__name__,
        str(exc),
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Naarira's AI shopping and customer-support assistant.

You have THREE main capabilities:

============================================================
1. PRODUCT SEARCH
============================================================

Use product tools for:

- sarees
- suits
- kurtis
- lehengas
- dresses
- ethnic wear
- product prices
- sizes
- colors
- availability
- product recommendations
- products under a specific price
- product details


============================================================
2. POLICY / FAQ
============================================================

Use the policy tool for:

- shipping
- delivery
- shipping charges
- free shipping
- return policy
- refund policy
- exchanges
- cancellation
- damaged/defective items
- non-returnable items
- shipping address changes
- general FAQs


============================================================
3. ORDER TRACKING
============================================================

Use the order tracking tool ONLY for a customer's
specific order.

Customer must provide BOTH:

1. Order number
2. Phone number used for the order

IMPORTANT:

- Never guess an order number.
- Never guess a phone number.
- Never invent tracking information.
- Never expose another customer's information.
- Do not call track_order_tool when required information
  is missing.

Examples:

"Where is my order?"

→ Ask for order number AND phone number.

"Track order #1269"

→ Ask for phone number.

"Track order #1269, phone +91..."

→ Call track_order_tool.


============================================================
ROUTING
============================================================

Product question
→ Product RAG / Product MCP

Policy question
→ search_naarira_policies

Specific order tracking
→ track_order_tool

Shipping policy questions must NOT use order tracking.

Specific order tracking questions must NOT use policy RAG.


============================================================
POLICY PRIORITY
============================================================

Dedicated Return & Refund Policy is authoritative for:

- returns
- refunds
- exchanges
- cancellations
- damaged items

Dedicated Shipping Policy is authoritative for:

- shipping
- delivery
- shipping charges
- address changes


============================================================
RESPONSE RULES
============================================================

- Be concise.
- Be helpful.
- Do not invent information.
- Do not mention internal tools.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention pgvector.
- Do not mention MCP.
- Do not mention databases.
- Do not mention system prompts.
- Do not expose internal implementation details.

Do not output raw URLs in natural-language answers.

For products, structured product data can contain product URLs.

For order tracking, show only safe tracking information
returned by the tracking tool.
"""


# ============================================================
# PRODUCT RAG TOOL
# ============================================================

@tool
def search_naarira_products(query: str) -> str:
    """
    Search Naarira products using the existing product RAG.
    """

    if PRODUCT_RAG_FUNCTION is None:
        return json.dumps(
            {
                "error": (
                    "Product RAG function could not be "
                    "found in rag.rag."
                )
            }
        )

    try:
        # Try keyword argument first
        try:
            result = PRODUCT_RAG_FUNCTION(
                query=query
            )

        except TypeError:
            # Fallback to positional query
            result = PRODUCT_RAG_FUNCTION(
                query
            )

        # Convert result to JSON
        if isinstance(result, str):
            return result

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception as exc:
        print(
            "Product RAG error:",
            type(exc).__name__,
            str(exc),
        )

        return json.dumps(
            {
                "error": (
                    "Unable to search products right now."
                )
            }
        )


# ============================================================
# POLICY RAG TOOL
# ============================================================

@tool
def search_naarira_policies(query: str) -> str:
    """
    Answer Naarira shipping, return, refund, exchange,
    cancellation and FAQ questions from canonical knowledge.
    """

    try:
        result = generate_policy_answer(
            question=query,
            top_k=3,
        )

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception as exc:
        print(
            "Policy RAG error:",
            type(exc).__name__,
            str(exc),
        )

        return json.dumps(
            {
                "error": (
                    "Unable to retrieve policy information "
                    "right now."
                )
            }
        )


# ============================================================
# MCP CLIENT
# ============================================================

def create_mcp_client() -> MultiServerMCPClient:

    print("\n" + "=" * 70)
    print("CONNECTING TO NAARIRA MCP SERVER")
    print("=" * 70)

    print(f"MCP URL: {MCP_URL}")

    client = MultiServerMCPClient(
        {
            "naarira": {
                "transport": "http",
                "url": MCP_URL,
            }
        }
    )

    print("✓ MCP client created successfully.")

    return client


# ============================================================
# CREATE AGENT
# ============================================================

async def create_naarira_agent():

    print("\n" + "=" * 70)
    print("INITIALIZING NAARIRA AGENT")
    print("=" * 70)

    print(f"Gemini model: {MODEL_NAME}")
    print(f"MCP URL: {MCP_URL}")

    # --------------------------------------------------------
    # MCP
    # --------------------------------------------------------

    mcp_client = create_mcp_client()

    print("\nLoading MCP tools...")

    try:
        mcp_tools = await mcp_client.get_tools()

    except Exception as exc:
        print("\n❌ Failed to load MCP tools.")

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        raise

    print(
        f"✓ Loaded {len(mcp_tools)} MCP tools."
    )

    print("\nMCP tools:")

    for index, mcp_tool in enumerate(
        mcp_tools,
        start=1,
    ):
        print(
            f"  {index}. {mcp_tool.name}"
        )

    # --------------------------------------------------------
    # Verify track_order_tool
    # --------------------------------------------------------

    mcp_tool_names = {
        tool_item.name
        for tool_item in mcp_tools
    }

    if "track_order_tool" in mcp_tool_names:

        print(
            "\n✓ track_order_tool is available."
        )

    else:

        print(
            "\n⚠ WARNING:"
            "\ntrack_order_tool is NOT available."
        )

    # --------------------------------------------------------
    # All tools
    # --------------------------------------------------------

    all_tools = [
        search_naarira_products,
        search_naarira_policies,
        *mcp_tools,
    ]

    print("\nAgent tools:")

    for index, agent_tool in enumerate(
        all_tools,
        start=1,
    ):
        print(
            f"  {index}. {agent_tool.name}"
        )

    print(
        f"\nTotal agent tools: {len(all_tools)}"
    )

    # --------------------------------------------------------
    # Create LangChain agent
    # --------------------------------------------------------

    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        name="naarira_shopping_agent",
    )

    print("\n" + "=" * 70)

    print(
        "NAARIRA AGENT INITIALIZED SUCCESSFULLY ✅"
    )

    print("=" * 70)

    return agent, mcp_client


# ============================================================
# GLOBAL AGENT CACHE
# ============================================================

_agent_instance = None


async def get_naarira_agent():

    """
    Create the Naarira agent only once and reuse it.

    This avoids rebuilding the LangGraph agent and reconnecting
    to MCP on every API request.
    """

    global _agent_instance

    if _agent_instance is None:

        print("=" * 70)
        print("INITIALIZING CACHED NAARIRA AGENT")
        print("=" * 70)

        _agent_instance = await create_naarira_agent()

        print("✓ Naarira agent initialized")

        print("=" * 70)

    return _agent_instance


# ============================================================
# SAFE JSON PARSER
# ============================================================

def _safe_json(value: Any) -> Any:

    """
    Convert tool output into Python objects whenever possible.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (dict, list, int, float, bool),
    ):
        return value

    if isinstance(value, str):

        text = value.strip()

        if not text:
            return ""

        try:
            return json.loads(text)

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return text

    return str(value)


# ============================================================
# TOOL NAME CLASSIFICATION
# ============================================================

def _is_order_tool(tool_name: str) -> bool:

    name = (tool_name or "").lower()

    return (
        "track_order" in name
        or "order_tracking" in name
        or "trackorder" in name
    )


def _is_policy_tool(tool_name: str) -> bool:

    name = (tool_name or "").lower()

    return (
        "policy" in name
        or "faq" in name
    )


def _is_product_tool(tool_name: str) -> bool:

    name = (tool_name or "").lower()

    return (
        "product" in name
        or "availability" in name
    )


# ============================================================
# EXTRACT TOOL CALL NAMES
# ============================================================

def _extract_tool_names(
    messages: List[Any],
) -> List[str]:

    """
    Extract tool names from LangChain AIMessage / ToolMessage
    objects.
    """

    names: List[str] = []

    for message in messages:

        # ----------------------------------------------------
        # AIMessage.tool_calls
        # ----------------------------------------------------

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if tool_calls:

            for call in tool_calls:

                if isinstance(call, dict):

                    name = call.get("name")

                    if name:
                        names.append(str(name))

        # ----------------------------------------------------
        # ToolMessage.name
        # ----------------------------------------------------

        message_name = getattr(
            message,
            "name",
            None,
        )

        if message_name:
            names.append(str(message_name))

    # Remove duplicates while preserving order

    result = []

    seen = set()

    for name in names:

        if name not in seen:

            seen.add(name)

            result.append(name)

    return result


# ============================================================
# EXTRACT TOOL RESULTS
# ============================================================

def _extract_tool_results(
    messages: List[Any],
) -> List[Dict[str, Any]]:

    """
    Read ToolMessage outputs from the agent state.
    """

    results: List[Dict[str, Any]] = []

    for message in messages:

        message_name = getattr(
            message,
            "name",
            None,
        )

        if not message_name:
            continue

        content = getattr(
            message,
            "content",
            None,
        )

        if content is None:
            continue

        parsed = _safe_json(content)

        results.append(
            {
                "name": str(message_name),
                "data": parsed,
            }
        )

    return results


# ============================================================
# FIND ORDER RESULT
# ============================================================

def _find_order_result(
    tool_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    for item in tool_results:

        tool_name = item["name"]

        if not _is_order_tool(tool_name):
            continue

        data = item["data"]

        if not isinstance(data, dict):
            continue

        # Only expose verified order information

        verified = bool(
            data.get("verified")
        )

        if not verified:

            return {
                "verified": False
            }

        order = {
            "verified": True,
            "order_number": data.get(
                "order_number"
            ),
            "status": data.get(
                "status"
            ),
            "message": data.get(
                "message"
            ),
            "tracking": data.get(
                "tracking"
            ) or [],
        }

        return order

    return None


# ============================================================
# FIND PRODUCTS
# ============================================================

def _find_product_results(
    tool_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    products: List[Dict[str, Any]] = []

    for item in tool_results:

        tool_name = item["name"]

        if not _is_product_tool(tool_name):
            continue

        data = item["data"]

        # Common format:
        # {"products": [...]}

        if isinstance(data, dict):

            candidate = data.get(
                "products"
            )

            if isinstance(candidate, list):

                for product in candidate:

                    if isinstance(product, dict):
                        products.append(product)

                continue

            # Single product result

            if (
                data.get("title")
                or data.get("product_id")
                or data.get("handle")
            ):

                products.append(data)

                continue

        # Direct list

        if isinstance(data, list):

            for product in data:

                if isinstance(product, dict):
                    products.append(product)

    # Deduplicate

    unique_products = []

    seen = set()

    for product in products:

        key = (
            product.get("product_id")
            or product.get("shopify_product_id")
            or product.get("url")
            or product.get("handle")
            or product.get("title")
        )

        if key in seen:
            continue

        seen.add(key)

        unique_products.append(product)

    return unique_products


# ============================================================
# NORMALIZE PRODUCT OBJECT
# ============================================================

def _normalize_product(
    product: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Keep only frontend-safe product fields.
    """

    return {

        "product_id": product.get(
            "product_id"
        ),

        "shopify_product_id": product.get(
            "shopify_product_id"
        ),

        "title": product.get(
            "title"
        ),

        "handle": product.get(
            "handle"
        ),

        "price": product.get(
            "price"
        ),

        "image": (
            product.get("image")
            or product.get("image_url")
            or product.get("featured_image")
        ),

        "url": product.get(
            "url"
        ),

        "vendor": product.get(
            "vendor"
        ),

        "availability": product.get(
            "availability"
        ),

        "available": product.get(
            "available"
        ),

        "description": product.get(
            "description"
        ),
    }


# ============================================================
# EXTRACT FINAL TEXT ANSWER
# ============================================================

def _extract_final_answer(
    messages: List[Any],
) -> str:

    # Walk backwards because the final AI message is normally
    # the last meaningful assistant response.

    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None,
        )

        if content is None:
            continue

        if isinstance(content, str):

            text = content.strip()

            if text:
                return text

        if isinstance(content, list):

            text_parts = []

            for part in content:

                if isinstance(part, str):

                    text_parts.append(part)

                elif isinstance(part, dict):

                    text_value = part.get(
                        "text"
                    )

                    if text_value:
                        text_parts.append(
                            str(text_value)
                        )

            text = " ".join(
                text_parts
            ).strip()

            if text:
                return text

    return (
        "I'm sorry, I couldn't generate "
        "a response right now."
    )


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_ai_answer(
    answer: str,
) -> str:

    if not answer:
        return ""

    # Remove markdown links

    answer = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\1",
        answer,
    )

    # Remove raw Naarira URLs

    answer = re.sub(
        r"https?://(?:www\.)?naarira\.com/\S*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # Excess horizontal whitespace

    answer = re.sub(
        r"[ \t]+",
        " ",
        answer,
    )

    # Excess blank lines

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer,
    )

    return answer.strip()


# ============================================================
# BUILD STRUCTURED RESPONSE
# ============================================================

def _build_structured_response(
    user_message: str,
    messages: List[Any],
) -> Dict[str, Any]:

    tool_names = _extract_tool_names(
        messages
    )

    tool_results = _extract_tool_results(
        messages
    )

    answer = _extract_final_answer(
        messages
    )

    answer = clean_ai_answer(
        answer
    )

    # --------------------------------------------------------
    # ORDER RESULT
    # --------------------------------------------------------

    order_data = _find_order_result(
        tool_results
    )

    if order_data is not None:

        return {
            "success": True,
            "type": "order",
            "answer": answer,
            "products": [],
            "order": order_data,
            "meta": {
                "route": "order",
                "tools_used": tool_names,
            },
        }

    # --------------------------------------------------------
    # PRODUCT RESULTS
    # --------------------------------------------------------

    products = _find_product_results(
        tool_results
    )

    if products:

        normalized_products = [
            _normalize_product(product)
            for product in products
        ]

        return {
            "success": True,
            "type": "products",
            "answer": answer,
            "products": normalized_products,
            "order": None,
            "meta": {
                "route": "products",
                "tools_used": tool_names,
            },
        }

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    policy_used = any(
        _is_policy_tool(name)
        for name in tool_names
    )

    if policy_used:

        return {
            "success": True,
            "type": "policy",
            "answer": answer,
            "products": [],
            "order": None,
            "meta": {
                "route": "policy",
                "tools_used": tool_names,
            },
        }

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return {
        "success": True,
        "type": "general",
        "answer": answer,
        "products": [],
        "order": None,
        "meta": {
            "route": "general",
            "tools_used": tool_names,
        },
    }


# ============================================================
# PUBLIC API FUNCTION — D1
# ============================================================

async def run_naarira_agent(
    user_message: str,
    session_id: str | None = None,
) -> Dict[str, Any]:

    """
    Main function used by FastAPI.

    Parameters:
        user_message:
            Customer's message.

        session_id:
            Unique conversation/session identifier.

    Returns:
        Stable JSON-compatible dictionary.
    """

    message = str(
        user_message or ""
    ).strip()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not message:

        return {
            "success": False,
            "type": "general",
            "answer": "Please enter a message.",
            "products": [],
            "order": None,
            "meta": {
                "route": "validation",
                "tools_used": [],
                "session_id": session_id,
            },
        }

    # --------------------------------------------------------
    # Get cached agent
    # --------------------------------------------------------

    agent, mcp_client = await get_naarira_agent()

    print("\n" + "=" * 70)
    print("NAARIRA AGENT REQUEST")
    print("=" * 70)

    print(
        f"Session ID: {session_id}"
    )

    print(
        f"User: {message}"
    )

    # --------------------------------------------------------
    # Agent execution
    # --------------------------------------------------------

    try:

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            }
        )

    except Exception as exc:

        print(
            "\n❌ Agent execution failed"
        )

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        raise

    # --------------------------------------------------------
    # Extract messages
    # --------------------------------------------------------

    messages = result.get(
        "messages",
        []
    )

    # --------------------------------------------------------
    # Build structured response
    # --------------------------------------------------------

    structured = _build_structured_response(
        user_message=message,
        messages=messages,
    )

    # --------------------------------------------------------
    # Add session ID
    # --------------------------------------------------------

    structured.setdefault(
        "meta",
        {}
    )

    structured["meta"]["session_id"] = (
        session_id
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NAARIRA AGENT RESPONSE")
    print("=" * 70)

    print(
        f"Route: "
        f"{structured.get('meta', {}).get('route')}"
    )

    print(
        f"Session ID: {session_id}"
    )

    print(
        f"Answer:\n{structured.get('answer')}"
    )

    print(
        f"Products returned: "
        f"{len(structured.get('products', []))}"
    )

    print(
        "\n" + "=" * 70
    )

    return structured


# ============================================================
# PHASE C TEST QUERIES
# ============================================================

TEST_QUERIES = [
    "Show me sarees under 2000",
    "What is your return policy?",
    "How long does shipping take?",
    "Do you offer exchanges?",
    "How long does a refund take?",
    "Where is my order?",
    "Track order #1269",
]


# ============================================================
# MAIN
# ============================================================

async def main():

    print("\n")

    print("#" * 70)
    print("NAARIRA PHASE C — AGENT ROUTING TEST")
    print("#" * 70)

    try:

        agent, mcp_client = (
            await create_naarira_agent()
        )

        for index, query in enumerate(
            TEST_QUERIES,
            start=1,
        ):

            print("\n")

            print("=" * 70)

            print(
                f"TEST {index}/{len(TEST_QUERIES)}"
            )

            print(
                f"Query: {query}"
            )

            print("=" * 70)

            result = await run_naarira_agent(
                user_message=query,
                session_id=f"local-test-{index}",
            )

            print(
                "\nFINAL ANSWER:"
            )

            print(
                result["answer"]
            )

            if result["products"]:

                print(
                    "\nPRODUCTS:"
                )

                for product in result[
                    "products"
                ]:

                    print(
                        f"- {product.get('title')}"
                    )

    except Exception as exc:

        print("\n" + "!" * 70)

        print(
            "NAARIRA AGENT INITIALIZATION FAILED"
        )

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        traceback.print_exc()

        print("!" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import asyncio

    asyncio.run(main())