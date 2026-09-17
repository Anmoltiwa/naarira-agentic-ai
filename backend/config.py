import os

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is missing in environment variables."
    )


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing in environment variables."
    )


# ============================================================
# SHOPIFY
# ============================================================

# Shopify credentials are optional for the AI backend.
#
# They are required only when running:
#
#     sync_products.py
#
# or another Shopify Admin API operation.

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

# ============================================================
# SHOPIFY VALIDATION
# ============================================================

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


# ============================================================
# HELPER
# ============================================================

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