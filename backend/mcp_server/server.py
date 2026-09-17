import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    search_products,
    get_product_details,
    check_product_availability,
    track_order,
)

load_dotenv()


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

HOST = os.getenv("MCP_HOST", "127.0.0.1")

PORT = int(
    os.getenv(
        "PORT",
        os.getenv("MCP_PORT", "8001"),
    )
)


# =========================================================
# MCP SERVER
# =========================================================

mcp = FastMCP(
    "Naarira Product Server",
)


# =========================================================
# TOOL 1: SEARCH PRODUCTS
# =========================================================

@mcp.tool()
def search_products_tool(
    query: Annotated[
        str,
        Field(
            description=(
                "Keyword or product name to search for, "
                "such as saree, georgette or designer."
            ),
        ),
    ] = "",
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            description="Maximum number of products to return.",
        ),
    ] = 5,
    min_price: Annotated[
        float | None,
        Field(
            description="Minimum product price in INR.",
        ),
    ] = None,
    max_price: Annotated[
        float | None,
        Field(
            description="Maximum product price in INR.",
        ),
    ] = None,
    color: Annotated[
        str | None,
        Field(
            description="Desired product color.",
        ),
    ] = None,
    size: Annotated[
        str | None,
        Field(
            description="Desired product size.",
        ),
    ] = None,
    available_only: Annotated[
        bool,
        Field(
            description=(
                "Return only currently available variants."
            ),
        ),
    ] = False,
    category: Annotated[
        str | None,
        Field(
            description=(
                "Product category such as saree, "
                "kurti, lehenga or dress."
            ),
        ),
    ] = None,
) -> list[dict]:
    """
    Search Naarira products using keywords and filters.
    """

    return search_products(
        query=query,
        limit=limit,
        min_price=min_price,
        max_price=max_price,
        color=color,
        size=size,
        available_only=available_only,
        category=category,
    )


# =========================================================
# TOOL 2: PRODUCT DETAILS
# =========================================================

@mcp.tool()
def get_product_details_tool(
    product_id: Annotated[
        int,
        Field(
            ge=1,
            description="Database ID of the Naarira product.",
        ),
    ],
) -> dict:
    """
    Get complete product details including variants,
    prices, sizes, colors, inventory and availability.
    """

    product = get_product_details(
        product_id=product_id,
    )

    if product is None:
        return {
            "error": (
                f"Product with ID {product_id} "
                "was not found."
            ),
        }

    return product


# =========================================================
# TOOL 3: PRODUCT AVAILABILITY
# =========================================================

@mcp.tool()
def check_product_availability_tool(
    product_id: Annotated[
        int,
        Field(
            ge=1,
            description="Database ID of the Naarira product.",
        ),
    ],
) -> dict:
    """
    Check available sizes, colors and inventory
    for a Naarira product.
    """

    result = check_product_availability(
        product_id=product_id,
    )

    if result is None:
        return {
            "error": "Product not found.",
        }

    return result


# =========================================================
# TOOL 4: ORDER TRACKING
# =========================================================

@mcp.tool()
def track_order_tool(
    order_number: Annotated[
        str,
        Field(
            description=(
                "Shopify order number, for example 1269 "
                "or #1269."
            ),
        ),
    ],
    phone: Annotated[
        str,
        Field(
            description=(
                "Phone number used when placing the order."
            ),
        ),
    ],
) -> dict:
    """
    Track a customer's Shopify order.

    Both the order number and phone number are required.
    The phone number is used to verify order ownership.
    """

    return track_order(
        order_number=order_number,
        phone=phone,
    )


# =========================================================
# START MCP SERVER
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STARTING NAARIRA MCP SERVER")
    print("=" * 70)

    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print("Transport: Streamable HTTP")
    print(f"MCP endpoint: http://{HOST}:{PORT}/mcp")

    try:
        registered_tools = [
            tool.name
            for tool in mcp._tool_manager.list_tools()
        ]

        print("Registered tools:")
        for tool_name in registered_tools:
            print(f"  - {tool_name}")

    except Exception as exc:
        print(
            "Could not list registered tools:",
            str(exc),
        )

    print("=" * 70)

    mcp.run(
        transport="streamable-http",
        mount_path="/mcp",
    )