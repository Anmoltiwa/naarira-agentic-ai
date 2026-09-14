import os
import json
import asyncio
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_mcp_adapters.client import (
    MultiServerMCPClient
)

from rag.search import semantic_search


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is missing in .env"
    )


# ============================================================
# CONFIGURATION
# ============================================================

MCP_URL = (
    "http://127.0.0.1:8001/mcp"
)

MODEL_NAME = (
    "gemini-3.8-flash"
)


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
)


# ============================================================
# RAG TOOL
# ============================================================

@tool
def search_naarira_products_with_rag(
    query: str,
    candidate_product_ids: Annotated[
        list[int] | None,
        Field(
            description=(
                "Optional list of product IDs returned "
                "by MCP structured search. If provided, "
                "RAG must rank only those candidate products."
            )
        )
    ] = None,
) -> str:
    """
    Semantic search for Naarira product discovery.

    Use this for:
    - recommendations
    - similar products
    - style preferences
    - occasion-based discovery
    - natural-language product searches

    For exact:
    - price
    - color
    - size
    - stock
    - availability

    use MCP structured product search first.

    For hybrid requests, MCP can provide candidate
    product IDs which are then semantically ranked
    by this RAG tool.
    """

    print(
        "\n" + "-" * 60
    )

    print(
        "[RAG TOOL]"
    )

    print(
        f"Query: {query}"
    )

    if candidate_product_ids:

        print(
            "Candidate IDs:",
            candidate_product_ids
        )

    try:

        results = semantic_search(
            query=query,
            top_k=5,
            candidate_product_ids=(
                candidate_product_ids
            ),
        )

    except Exception as e:

        print(
            "[RAG ERROR]"
        )

        print(
            f"{type(e).__name__}: {str(e)}"
        )

        return json.dumps(
            {
                "error": (
                    "RAG search failed."
                )
            },
            ensure_ascii=False
        )

    if not results:

        return json.dumps(
            {
                "message": (
                    "No relevant Naarira products "
                    "were found."
                )
            },
            ensure_ascii=False
        )

    products = []

    for product in results:

        handle = product.get(
            "handle"
        )

        products.append({
            "product_id": product.get(
                "product_id"
            ),
            "title": product.get(
                "title"
            ),
            "category": (
                product.get(
                    "category"
                )
                or product.get(
                    "product_type"
                )
            ),
            "similarity": round(
                float(
                    product.get(
                        "similarity",
                        0
                    )
                ),
                4
            ),
            "content": product.get(
                "content",
                ""
            ),
            "handle": handle,
            "url": (
                f"https://naarira.com/products/"
                f"{handle}"
                if handle
                else None
            ),
        })

    print(
        f"[RAG TOOL] "
        f"Returned {len(products)} products"
    )

    return json.dumps(
        products,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Naarira's AI shopping agent.

Naarira is a women's ethnic fashion store.

Your job is to understand the customer's intent,
use the correct capability, retrieve real Naarira
data, and provide a grounded answer.

============================================================
AVAILABLE CAPABILITIES
============================================================

1. RAG SEMANTIC SEARCH

Tool:
search_naarira_products_with_rag

Use RAG for:

- recommendations
- similar products
- style preferences
- occasion-based discovery
- natural-language product discovery

Examples:

"Show me elegant sarees"

"I need something festive for a wedding"

"Show me something similar to this saree"

------------------------------------------------------------

2. MCP STRUCTURED SEARCH

Tool:
search_products_tool

Use MCP structured search for hard constraints:

- category
- minimum price
- maximum price
- color
- size
- stock
- availability

Examples:

"red sarees under ₹1500"

"blue sarees in size M"

"cotton sarees below ₹2000"

"designer sarees under ₹1000 in stock"

------------------------------------------------------------

3. PRODUCT DETAILS

Tool:
get_product_details_tool

Use for:

- exact product information
- product description
- variants
- prices
- sizes
- colors
- specifications

Example:

"Tell me about product 2"

------------------------------------------------------------

4. AVAILABILITY

Tool:
check_product_availability_tool

Use for:

- current availability
- stock
- inventory
- available sizes
- available colors

Example:

"Is product 2 available?"

============================================================
HARD CONSTRAINT EXTRACTION
============================================================

When the user provides exact constraints,
extract them and pass them to MCP.

Example:

"red sarees under ₹1500"

Use:

query = "saree"
color = "red"
max_price = 1500

Example:

"blue designer sarees in size M under ₹2000"

Use:

query = "designer saree"
color = "blue"
size = "M"
max_price = 2000

Example:

"cotton sarees under ₹1200 in stock"

Use:

query = "cotton saree"
max_price = 1200
available_only = true

Example:

"sarees between ₹1000 and ₹2000"

Use:

query = "saree"
min_price = 1000
max_price = 2000

============================================================
HYBRID RETRIEVAL
============================================================

Some requests contain BOTH:

1. semantic preferences
2. hard structured constraints

Example:

"Find me an elegant red saree for a wedding under ₹1500"

Hard constraints:

- category = saree
- color = red
- max_price = 1500

Semantic preferences:

- elegant
- wedding

For this type of query:

STEP 1:
Use search_products_tool first.

STEP 2:
Request a larger candidate pool when useful,
up to 20 products.

STEP 3:
Read the returned product IDs.

STEP 4:
Call search_naarira_products_with_rag using
the semantic part of the request and pass the
candidate product IDs.

STEP 5:
RAG must rank ONLY those candidates.

STEP 6:
If availability is explicitly requested,
use check_product_availability_tool.

IMPORTANT:

Never perform unrestricted RAG search when
hard constraints have been provided.

Hard constraints must be enforced by MCP.

============================================================
MULTI-TOOL REQUESTS
============================================================

Example:

"Find me an elegant red saree under ₹1500
and tell me whether it is available."

Use:

1. search_products_tool
2. RAG ranking when useful
3. check_product_availability_tool
4. final answer

Use only the tools necessary.

============================================================
GROUNDING
============================================================

Never invent:

- product names
- prices
- sizes
- colors
- availability
- stock
- inventory
- fabric
- specifications
- discounts
- product URLs

Tool results are the source of truth.

Never combine attributes from unrelated products.

If requested information is unavailable,
say so clearly.

============================================================
PRODUCT URL
============================================================

Use the exact product URL returned by a tool.

Never invent a URL.

Do not generate Markdown product links
in the answer.

Product URLs are returned separately in
the products[] response for the frontend.

============================================================
FINAL ANSWER
============================================================

Be concise and customer-friendly.

When multiple products match:

- mention product names
- include relevant verified details
- don't invent missing information

Do not mention internal concepts such as:

- RAG
- MCP
- LangGraph
- pgvector
- embeddings
- prompts
- database
- tools

unless the customer explicitly asks about
the technical architecture.

============================================================
AGENT BEHAVIOR
============================================================

You are an agent.

Do not immediately answer Naarira-specific
product questions from model knowledge.

Use the appropriate capability.

Process:

Understand intent
→ extract constraints
→ choose capability
→ call tool
→ inspect result
→ call another tool if necessary
→ answer using verified data.

For price, color, size, category and availability,
prefer structured MCP filtering.

For semantic preferences, use RAG.

For exact product details, use MCP.

For availability, use MCP.
"""


# ============================================================
# CREATE AGENT
# ============================================================

async def create_naarira_agent():

    print(
        "\nConnecting to Naarira MCP server..."
    )

    # --------------------------------------------------------
    # MCP client
    # --------------------------------------------------------

    mcp_client = MultiServerMCPClient(
        {
            "naarira": {
                "transport": "http",
                "url": MCP_URL,
            }
        }
    )

    # --------------------------------------------------------
    # Load MCP tools
    # --------------------------------------------------------

    mcp_tools = await (
        mcp_client.get_tools()
    )

    print(
        "\nMCP tools loaded:"
    )

    for mcp_tool in mcp_tools:

        print(
            f"  - {mcp_tool.name}"
        )

    # --------------------------------------------------------
    # Combine RAG + MCP
    # --------------------------------------------------------

    all_tools = [
        search_naarira_products_with_rag,
        *mcp_tools
    ]

    print(
        "\nAgent tools:"
    )

    for agent_tool in all_tools:

        print(
            f"  - {agent_tool.name}"
        )

    # --------------------------------------------------------
    # Create LangGraph agent
    # --------------------------------------------------------

    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        name="naarira_shopping_agent",
    )

    return agent


# ============================================================
# EXTRACT FINAL RESPONSE
# ============================================================

def extract_response(
    result: dict
) -> str:
    """
    Extract final AI text from agent result.
    """

    messages = result.get(
        "messages",
        []
    )

    if not messages:

        return "No response generated."

    # --------------------------------------------------------
    # Search backwards for final AI message
    # --------------------------------------------------------

    for message in reversed(
        messages
    ):

        message_type = getattr(
            message,
            "type",
            None
        )

        if message_type != "ai":

            continue

        content = getattr(
            message,
            "content",
            ""
        )

        # ----------------------------------------------------
        # Plain string
        # ----------------------------------------------------

        if isinstance(
            content,
            str
        ):

            return content.strip()

        # ----------------------------------------------------
        # Structured content
        # ----------------------------------------------------

        if isinstance(
            content,
            list
        ):

            text_parts = []

            for block in content:

                if isinstance(
                    block,
                    dict
                ):

                    if block.get(
                        "type"
                    ) == "text":

                        text = block.get(
                            "text",
                            ""
                        )

                        if text:

                            text_parts.append(
                                text
                            )

                elif isinstance(
                    block,
                    str
                ):

                    text_parts.append(
                        block
                    )

            if text_parts:

                return "\n".join(
                    text_parts
                ).strip()

    return "No response generated."


# ============================================================
# JSON OBJECT EXTRACTION
# ============================================================

def _extract_json_objects(
    value: Any
) -> list[dict]:
    """
    Recursively extract product-like dictionaries
    from tool outputs.
    """

    found = []

    if value is None:

        return found

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        if (
            value.get("title")
            and value.get("url")
        ):

            found.append(
                value
            )

        for nested_value in value.values():

            found.extend(
                _extract_json_objects(
                    nested_value
                )
            )

        return found

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        value,
        list
    ):

        for item in value:

            found.extend(
                _extract_json_objects(
                    item
                )
            )

        return found

    # --------------------------------------------------------
    # JSON string
    # --------------------------------------------------------

    if isinstance(
        value,
        str
    ):

        text_value = value.strip()

        if not text_value:

            return found

        try:

            parsed = json.loads(
                text_value
            )

            found.extend(
                _extract_json_objects(
                    parsed
                )
            )

        except json.JSONDecodeError:

            # Normal text — ignore
            pass

    return found


# ============================================================
# EXTRACT CLICKABLE PRODUCTS
# ============================================================

def extract_products_from_result(
    result: dict
) -> list[dict]:
    """
    Extract product information from RAG/MCP tool
    messages so the frontend can create clickable
    product cards.
    """

    messages = result.get(
        "messages",
        []
    )

    products = []

    seen_ids = set()
    seen_urls = set()

    def add_product(
        product: dict
    ):

        if not isinstance(
            product,
            dict
        ):

            return

        title = product.get(
            "title"
        )

        url = product.get(
            "url"
        )

        if not title or not url:

            return

        product_id = product.get(
            "product_id"
        )

        # ----------------------------------------------------
        # Deduplicate by ID
        # ----------------------------------------------------

        if product_id is not None:

            if product_id in seen_ids:

                return

            seen_ids.add(
                product_id
            )

        # ----------------------------------------------------
        # Deduplicate by URL
        # ----------------------------------------------------

        if url in seen_urls:

            return

        seen_urls.add(
            url
        )

        products.append({
            "product_id": product_id,
            "title": title,
            "handle": product.get(
                "handle"
            ),
            "category": (
                product.get(
                    "category"
                )
                or product.get(
                    "product_type"
                )
            ),
            "url": url,
        })

    # --------------------------------------------------------
    # Inspect tool messages
    # --------------------------------------------------------

    for message in messages:

        message_type = getattr(
            message,
            "type",
            None
        )

        if message_type != "tool":

            continue

        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        content = getattr(
            message,
            "content",
            None
        )

        extracted = _extract_json_objects(
            content
        )

        for product in extracted:

            add_product(
                product
            )

        # ----------------------------------------------------
        # Artifact / structured output
        # ----------------------------------------------------

        artifact = getattr(
            message,
            "artifact",
            None
        )

        if artifact is not None:

            extracted = _extract_json_objects(
                artifact
            )

            for product in extracted:

                add_product(
                    product
                )

    return products


# ============================================================
# PRINT AGENT TRACE
# ============================================================

def print_agent_trace(
    result: dict
):
    """
    Print the tools selected and called by
    the agent.

    Useful for development and interviews.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "AGENT TOOL TRACE"
    )

    print(
        "=" * 60
    )

    messages = result.get(
        "messages",
        []
    )

    for message in messages:

        message_type = getattr(
            message,
            "type",
            None
        )

        # ----------------------------------------------------
        # AI message with tool calls
        # ----------------------------------------------------

        if message_type == "ai":

            tool_calls = getattr(
                message,
                "tool_calls",
                []
            )

            for call in tool_calls:

                print(
                    "\nTool:"
                )

                print(
                    f"  {call.get('name')}"
                )

                print(
                    "Arguments:"
                )

                print(
                    json.dumps(
                        call.get(
                            "args",
                            {}
                        ),
                        ensure_ascii=False,
                        indent=2
                    )
                )

        # ----------------------------------------------------
        # Tool result
        # ----------------------------------------------------

        elif message_type == "tool":

            tool_name = getattr(
                message,
                "name",
                "unknown"
            )

            print(
                f"\nTool result: {tool_name}"
            )

    print(
        "\n" + "=" * 60
    )


# ============================================================
# RUN AGENT
# ============================================================

async def run_naarira_agent(
    query: str,
    agent=None,
    debug: bool = False
) -> dict:
    """
    Run the Naarira agent for one query.

    Returns:

    {
        "answer": str,
        "products": list
    }
    """

    if not query or not query.strip():

        return {
            "answer": (
                "Please enter a question."
            ),
            "products": []
        }

    # --------------------------------------------------------
    # Reuse agent if supplied
    # --------------------------------------------------------

    if agent is None:

        agent = await (
            create_naarira_agent()
        )

    print(
        f"\n[USER QUERY] {query}"
    )

    # --------------------------------------------------------
    # Invoke agent
    # --------------------------------------------------------

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query.strip()
                }
            ]
        }
    )

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    if debug:

        print_agent_trace(
            result
        )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    answer = extract_response(
        result
    )

    # --------------------------------------------------------
    # Product cards
    # --------------------------------------------------------

    products = extract_products_from_result(
        result
    )

    return {
        "answer": answer,
        "products": products
    }


# ============================================================
# LOCAL CLI TEST
# ============================================================

async def main():

    print(
        "=" * 60
    )

    print(
        "NAARIRA AGENT"
    )

    print(
        "HYBRID RAG + MCP"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Create once
    # --------------------------------------------------------

    agent = await (
        create_naarira_agent()
    )

    print(
        "\nAgent ready."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    # --------------------------------------------------------
    # Interactive loop
    # --------------------------------------------------------

    while True:

        query = input(
            "\nYou: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nAgent stopped."
            )

            break

        if not query:

            continue

        try:

            result = await (
                run_naarira_agent(
                    query=query,
                    agent=agent,
                    debug=True
                )
            )

            print(
                "\nNaarira AI:"
            )

            print(
                result["answer"]
            )

            # ------------------------------------------------
            # Clickable products for frontend
            # ------------------------------------------------

            if result["products"]:

                print(
                    "\nClickable Products:"
                )

                for product in result[
                    "products"
                ]:

                    print(
                        f"\n- {product['title']}"
                    )

                    print(
                        f"  URL: "
                        f"{product['url']}"
                    )

            else:

                print(
                    "\nNo product cards extracted."
                )

        except Exception as e:

            print(
                "\nAGENT ERROR:"
            )

            print(
                f"{type(e).__name__}: {str(e)}"
            )

        print(
            "\n" + "-" * 60
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )