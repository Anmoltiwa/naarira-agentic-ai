import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    search_products,
    get_product_details,
    check_product_availability,
)

load_dotenv()


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

HOST = os.getenv("MCP_HOST", "127.0.0.1")

PORT = int(
    os.getenv(
        "PORT",
        os.getenv("MCP_PORT", "8001")
    )
)


# ---------------------------------------------------------
# MCP Server
# ---------------------------------------------------------

mcp = FastMCP(
    "Naarira Product Server",
    host=HOST,
    port=PORT,
)


# ---------------------------------------------------------
# TOOL 1: Search Products
# ---------------------------------------------------------

@mcp.tool()
def search_products_tool(
    query: Annotated[
        str,
        Field(
            description=(
                "Keyword or product name to search for, "
                "such as saree, georgette or designer."
            )
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
            description="Return only currently available variants.",
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
    Search Naarira's product catalog using
    keywords and structured filters such as
    price, color, size, category and availability.
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


# ---------------------------------------------------------
# TOOL 2: Product Details
# ---------------------------------------------------------

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
    Get complete details of a Naarira product,
    including variants, prices, sizes, colors,
    inventory and availability.
    """

    product = get_product_details(product_id=product_id)

    if product is None:
        return {
            "error": (
                f"Product with ID {product_id} "
                "was not found."
            )
        }

    return product


# ---------------------------------------------------------
# TOOL 3: Product Availability
# ---------------------------------------------------------

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
        product_id=product_id
    )

    if result is None:
        return {
            "error": "Product not found."
        }

    return result


# ---------------------------------------------------------
# Start MCP Server
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("STARTING NAARIRA MCP SERVER")
    print("=" * 60)

    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print("Transport: Streamable HTTP")
    print(f"MCP endpoint: http://{HOST}:{PORT}/mcp")

    print(
        "Registered tools:",
        [
            tool.name
            for tool in mcp._tool_manager.list_tools()
        ],
    )

    print("=" * 60)

    mcp.run(
        transport="streamable-http",
        mount_path="/mcp",
    )