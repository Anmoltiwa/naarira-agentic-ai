import os
from dotenv import load_dotenv

load_dotenv()


SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP")
SHOPIFY_CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
SHOPIFY_CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

DATABASE_URL = os.getenv("DATABASE_URL")


if not SHOPIFY_SHOP:
    raise ValueError("SHOPIFY_SHOP is missing in .env")

if not SHOPIFY_CLIENT_ID:
    raise ValueError("SHOPIFY_CLIENT_ID is missing in .env")

if not SHOPIFY_CLIENT_SECRET:
    raise ValueError("SHOPIFY_CLIENT_SECRET is missing in .env")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing in .env")


SHOPIFY_API_VERSION = "2026-07"

SHOPIFY_GRAPHQL_URL = (
    f"https://{SHOPIFY_SHOP}/admin/api/"
    f"{SHOPIFY_API_VERSION}/graphql.json"
)