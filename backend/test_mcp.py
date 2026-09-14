import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "http://127.0.0.1:8001/mcp"


async def main():

    print("=" * 60)
    print("NAARIRA MCP CLIENT TEST")
    print("=" * 60)

    print(f"\nConnecting to: {MCP_URL}")

    async with streamable_http_client(MCP_URL) as (
        read_stream,
        write_stream,
        _
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            print("\nInitializing MCP session...")

            await session.initialize()

            print(
                "MCP session initialized successfully."
            )

            # ------------------------------------------------
            # LIST TOOLS
            # ------------------------------------------------

            print("\nListing available tools...")

            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:

                print(f"- {tool.name}")

                if tool.description:
                    print(
                        f"  Description: "
                        f"{tool.description}"
                    )

            # ------------------------------------------------
            # TEST SEARCH
            # ------------------------------------------------

            print("\n" + "-" * 60)
            print("TEST 1: search_products_tool")
            print("-" * 60)

            search_result = await session.call_tool(
                "search_products_tool",
                {
                    "query": "saree",
                    "limit": 5
                }
            )

            print("\nSearch result:")

            for content in search_result.content:

                if hasattr(content, "text"):
                    print(content.text)

            # ------------------------------------------------
            # TEST PRODUCT DETAILS
            # ------------------------------------------------

            print("\n" + "-" * 60)
            print("TEST 2: get_product_details_tool")
            print("-" * 60)

            details_result = await session.call_tool(
                "get_product_details_tool",
                {
                    "product_id": 2
                }
            )

            print("\nProduct details:")

            for content in details_result.content:

                if hasattr(content, "text"):
                    print(content.text)

            # ------------------------------------------------
            # TEST PRODUCT AVAILABILITY
            # ------------------------------------------------

            print("\n" + "-" * 60)
            print("TEST 3: check_product_availability_tool")
            print("-" * 60)

            availability_result = await session.call_tool(
                "check_product_availability_tool",
                {
                    "product_id": 2
                }
            )

            print("\nProduct availability:")

            for content in availability_result.content:

                if hasattr(content, "text"):
                    print(content.text)

            # ------------------------------------------------
            # TEST PRODUCT AVAILABILITY WITH SIZE + COLOR
            # ------------------------------------------------

            print("\n" + "-" * 60)
            print("TEST 4: availability with size + color")
            print("-" * 60)

            availability_variant_result = await session.call_tool(
                "check_product_availability_tool",
                {
                    "product_id": 2,
                    "size": "M",
                    "color": "Red"
                }
            )

            print("\nSpecific variant availability:")

            for content in availability_variant_result.content:

                if hasattr(content, "text"):
                    print(content.text)

    print("\n" + "=" * 60)
    print("MCP TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    asyncio.run(main())