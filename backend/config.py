import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL"
)
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is missing in environment variables."
    )
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing in environment variables."
    )
SHOPIFY_SHOP = os.getenv(
    "SHOPIFY_SHOP"
)
SHOPIFY_CLIENT_ID = os.getenv(
    "SHOPIFY_CLIENT_ID"
)
SHOPIFY_CLIENT_SECRET = os.getenv(
    "SHOPIFY_CLIENT_SECRET"
)
SHOPIFY_API_VERSION = os.getenv(
    "SHOPIFY_API_VERSION",
    "2026-01",
)
SHOPIFY_GRAPHQL_URL = (
    f"https://{SHOPIFY_SHOP}"
    f"/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
)
def validate_shopify_config():
    """
    Validate Shopify credentials only when a Shopify
    operation is actually required.
    """

    missing = []

    if not SHOPIFY_SHOP:
        missing.append(
            "SHOPIFY_SHOP"
        )

    if not SHOPIFY_CLIENT_ID:
        missing.append(
            "SHOPIFY_CLIENT_ID"
        )

    if not SHOPIFY_CLIENT_SECRET:
        missing.append(
            "SHOPIFY_CLIENT_SECRET"
        )

    if missing:

        raise ValueError(
            "Missing Shopify environment variables: "
            + ", ".join(missing)
        )
def is_shopify_configured():
    """
    Return True when all Shopify credentials are available.
    """

    return all(
        [
            SHOPIFY_SHOP,
            SHOPIFY_CLIENT_ID,
            SHOPIFY_CLIENT_SECRET,
        ]
    )