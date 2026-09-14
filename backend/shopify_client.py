import requests

from config import (
    SHOPIFY_SHOP,
    SHOPIFY_CLIENT_ID,
    SHOPIFY_CLIENT_SECRET,
    SHOPIFY_GRAPHQL_URL,
)


def get_access_token():
    """
    Get a temporary Shopify Admin API access token
    using the Client Credentials Grant.
    """

    url = f"https://{SHOPIFY_SHOP}/admin/oauth/access_token"

    payload = {
        "grant_type": "client_credentials",
        "client_id": SHOPIFY_CLIENT_ID,
        "client_secret": SHOPIFY_CLIENT_SECRET,
    }

    response = requests.post(
        url,
        data=payload,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    access_token = data.get("access_token")

    if not access_token:
        raise RuntimeError(
            f"Shopify did not return an access token: {data}"
        )

    return access_token


def graphql_request(access_token, query, variables=None):
    """
    Execute a Shopify Admin GraphQL API request.
    """

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }

    payload = {
        "query": query,
        "variables": variables or {},
    }

    response = requests.post(
        SHOPIFY_GRAPHQL_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "errors" in data:
        raise RuntimeError(
            f"Shopify GraphQL error: {data['errors']}"
        )

    return data


PRODUCTS_QUERY = """
query GetProducts($first: Int!, $after: String) {
  products(first: $first, after: $after) {
    edges {
      cursor
      node {
        id
        title
        handle
        description
        vendor
        productType
        status
        tags
        createdAt
        updatedAt

        featuredImage {
          url
        }

        variants(first: 250) {
          edges {
            node {
              id
              title
              sku
              price
              compareAtPrice
              inventoryQuantity
              availableForSale
              createdAt
              updatedAt

              selectedOptions {
                name
                value
              }

              image {
                url
              }
            }
          }

          pageInfo {
            hasNextPage
            endCursor
          }
        }
      }
    }

    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


def fetch_all_products(access_token):
    """
    Fetch all Shopify products using cursor pagination.
    """

    products = []

    cursor = None

    while True:

        variables = {
            "first": 250,
            "after": cursor,
        }

        data = graphql_request(
            access_token,
            PRODUCTS_QUERY,
            variables,
        )

        products_data = data["data"]["products"]

        for edge in products_data["edges"]:
            products.append(edge["node"])

        page_info = products_data["pageInfo"]

        print(
            f"Fetched {len(products)} products..."
        )

        if not page_info["hasNextPage"]:
            break

        cursor = page_info["endCursor"]

    return products