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
def get_order_for_tracking(
    order_number: str,
    phone: str,
) -> dict | None:
    """
    Find and verify a Shopify order using:

        Order Number + Phone Number

    Shopify order search supports the `name:` filter.
    Phone is verified locally against the returned Order.phone.
    """

    order_number = normalize_order_number(order_number)
    phone = phone.strip()

    if not order_number:
        raise ValueError(
            "Order number is required."
        )

    if not phone:
        raise ValueError(
            "Phone number is required."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Shopify's orders query does NOT provide a phone:
    # search filter in the documented filter list.
    #
    # Therefore:
    # 1. Search by order name
    # 2. Retrieve order.phone
    # 3. Verify phone locally
    # --------------------------------------------------------

    search_query = f"name:{order_number}"

    access_token = get_access_token()

    data = graphql_request(
        access_token,
        ORDER_TRACKING_QUERY,
        {
            "first": 10,
            "query": search_query,
        },
    )

    orders_data = (
        data
        .get("data", {})
        .get("orders", {})
    )

    edges = orders_data.get(
        "edges",
        [],
    )

    if not edges:
        return None

    # --------------------------------------------------------
    # Verify order number + phone
    # --------------------------------------------------------

    matched_order = None

    for edge in edges:

        order = edge["node"]

        shopify_name = (
            order.get("name")
            or ""
        )

        order_phone = (
            order.get("phone")
            or ""
        )

        normalized_shopify_number = (
            normalize_order_number(
                shopify_name
            )
        )

        if normalized_shopify_number != order_number:
            continue

        if not phone_matches(
            phone,
            order_phone,
        ):
            continue

        matched_order = order
        break

    if not matched_order:
        return None

    # --------------------------------------------------------
    # Fulfillment information
    # --------------------------------------------------------

    fulfillments = []

    for fulfillment in (
        matched_order.get(
            "fulfillments"
        )
        or []
    ):

        tracking_items = []

        for tracking in (
            fulfillment.get(
                "trackingInfo"
            )
            or []
        ):

            tracking_items.append(
                {
                    "company": tracking.get(
                        "company"
                    ),
                    "number": tracking.get(
                        "number"
                    ),
                    "url": tracking.get(
                        "url"
                    ),
                }
            )

        fulfillments.append(
            {
                "status": fulfillment.get(
                    "status"
                ),
                "created_at": fulfillment.get(
                    "createdAt"
                ),
                "updated_at": fulfillment.get(
                    "updatedAt"
                ),
                "delivered_at": fulfillment.get(
                    "deliveredAt"
                ),
                "estimated_delivery_at": (
                    fulfillment.get(
                        "estimatedDeliveryAt"
                    )
                ),
                "in_transit_at": fulfillment.get(
                    "inTransitAt"
                ),
                "tracking": tracking_items,
            }
        )

    # --------------------------------------------------------
    # Safe result
    # --------------------------------------------------------

    return {
        "verified": True,
        "order_number": matched_order.get(
            "name"
        ),
        "order_status": matched_order.get(
            "displayFulfillmentStatus"
        ),
        "financial_status": matched_order.get(
            "displayFinancialStatus"
        ),
        "created_at": matched_order.get(
            "createdAt"
        ),
        "updated_at": matched_order.get(
            "updatedAt"
        ),
        "fulfillments": fulfillments,
    }
# ============================================================
# ORDER TRACKING
# ============================================================

ORDER_TRACKING_QUERY = """
query GetOrderForTracking($first: Int!, $query: String!) {
  orders(
    first: $first,
    query: $query,
    sortKey: UPDATED_AT,
    reverse: true
  ) {
    edges {
      node {
        id
        name
        phone
        createdAt
        updatedAt

        displayFinancialStatus
        displayFulfillmentStatus

        fulfillments(first: 10) {
          id
          status
          createdAt
          updatedAt
          deliveredAt
          estimatedDeliveryAt
          inTransitAt

          trackingInfo(first: 10) {
            company
            number
            url
          }
        }
      }
    }
  }
}
"""


# ============================================================
# NORMALIZE ORDER NUMBER
# ============================================================

def normalize_order_number(
    order_number: str,
) -> str:
    """
    Convert different order-number formats into
    a simple numeric/string value.

    Examples:

        1001
        #1001
        Order #1001

    become:

        1001
    """

    if order_number is None:
        return ""

    value = str(
        order_number
    ).strip()

    value = value.replace(
        "Order",
        "",
    )

    value = value.replace(
        "order",
        "",
    )

    value = value.replace(
        "#",
        "",
    )

    return value.strip()


# ============================================================
# NORMALIZE PHONE
# ============================================================

def normalize_phone(
    phone: str,
) -> str:
    """
    Keep digits only.

    Example:

        +91 98765 43210
        919876543210

    become digit-only strings.
    """

    if phone is None:
        return ""

    return "".join(
        character
        for character in str(phone)
        if character.isdigit()
    )


# ============================================================
# PHONE MATCHING
# ============================================================

def phone_matches(
    customer_phone: str,
    order_phone: str,
) -> bool:
    """
    Compare customer-provided phone with Shopify Order.phone.

    Exact match is checked first.

    Then the last 10 digits are compared, which handles
    common Indian +91 country-code formatting differences.
    """

    customer_digits = normalize_phone(
        customer_phone
    )

    order_digits = normalize_phone(
        order_phone
    )

    if not customer_digits:
        return False

    if not order_digits:
        return False

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if customer_digits == order_digits:
        return True

    # --------------------------------------------------------
    # Last 10 digits
    # --------------------------------------------------------

    if (
        len(customer_digits) >= 10
        and len(order_digits) >= 10
    ):

        return (
            customer_digits[-10:]
            == order_digits[-10:]
        )

    return False


def get_order_for_tracking(
    order_number: str,
    phone: str,
) -> dict | None:
    """
    Debug version of Shopify order lookup.
    Searches by order name and prints returned order
    number + phone for verification.
    """

    order_number = normalize_order_number(
        order_number
    )

    phone = str(
        phone or ""
    ).strip()

    if not order_number:
        raise ValueError(
            "Order number is required."
        )

    if not phone:
        raise ValueError(
            "Phone number is required."
        )

    search_query = f"name:{order_number}"

    print("\n" + "=" * 70)
    print("SHOPIFY ORDER DEBUG")
    print("=" * 70)

    print(
        f"Requested order number: {order_number}"
    )

    print(
        "Requested phone: **********"
    )

    print(
        f"Shopify search query: {search_query}"
    )

    # --------------------------------------------------------
    # Shopify API
    # --------------------------------------------------------

    access_token = get_access_token()

    data = graphql_request(
        access_token,
        ORDER_TRACKING_QUERY,
        {
            "first": 10,
            "query": search_query,
        },
    )

    # --------------------------------------------------------
    # GraphQL response
    # --------------------------------------------------------

    orders_data = (
        data
        .get("data", {})
        .get("orders", {})
    )

    edges = orders_data.get(
        "edges",
        [],
    )

    print(
        f"\nOrders returned by Shopify: {len(edges)}"
    )

    if not edges:

        print(
            "\n❌ Shopify returned ZERO orders."
        )

        print(
            "This means the order number/search query "
            "needs investigation."
        )

        print("=" * 70)

        return None

    # --------------------------------------------------------
    # Inspect returned orders
    # --------------------------------------------------------

    for index, edge in enumerate(
        edges,
        start=1,
    ):

        order = edge.get(
            "node"
        )

        if not order:
            continue

        shopify_name = (
            order.get("name")
            or ""
        )

        shopify_phone = (
            order.get("phone")
            or ""
        )

        print(
            f"\nORDER {index}"
        )

        print(
            f"Shopify order name: "
            f"{shopify_name}"
        )

        print(
            f"Shopify phone exists: "
            f"{bool(shopify_phone)}"
        )

        print(
            f"Shopify phone length: "
            f"{len(normalize_phone(shopify_phone))}"
        )

        print(
            f"Fulfillment status: "
            f"{order.get('displayFulfillmentStatus')}"
        )

        print(
            f"Financial status: "
            f"{order.get('displayFinancialStatus')}"
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    matched_order = None

    for edge in edges:

        order = edge.get(
            "node"
        )

        if not order:
            continue

        shopify_name = (
            order.get("name")
            or ""
        )

        shopify_phone = (
            order.get("phone")
            or ""
        )

        normalized_shopify_number = (
            normalize_order_number(
                shopify_name
            )
        )

        print(
            "\nVerification:"
        )

        print(
            f"Requested order: "
            f"{order_number}"
        )

        print(
            f"Shopify order: "
            f"{normalized_shopify_number}"
        )

        print(
            f"Order number match: "
            f"{normalized_shopify_number == order_number}"
        )

        print(
            f"Phone match: "
            f"{phone_matches(phone, shopify_phone)}"
        )

        if (
            normalized_shopify_number
            == order_number
            and phone_matches(
                phone,
                shopify_phone,
            )
        ):

            matched_order = order

            break

    # --------------------------------------------------------
    # Verification failed
    # --------------------------------------------------------

    if not matched_order:

        print(
            "\n❌ Order verification failed."
        )

        print(
            "Possible reasons:"
        )

        print(
            "1. Order number does not exist."
        )

        print(
            "2. Shopify order number has a different format."
        )

        print(
            "3. Shopify order has no phone number."
        )

        print(
            "4. Provided phone does not match."
        )

        print("=" * 70)

        return None

    print(
        "\n✅ ORDER VERIFIED"
    )

    # ========================================================
    # FULFILLMENTS
    # ========================================================

    fulfillments = []

    for fulfillment in (
        matched_order.get(
            "fulfillments"
        )
        or []
    ):

        tracking_items = []

        for tracking in (
            fulfillment.get(
                "trackingInfo"
            )
            or []
        ):

            tracking_items.append(
                {
                    "company": tracking.get(
                        "company"
                    ),
                    "number": tracking.get(
                        "number"
                    ),
                    "url": tracking.get(
                        "url"
                    ),
                }
            )

        fulfillments.append(
            {
                "status": fulfillment.get(
                    "status"
                ),
                "created_at": fulfillment.get(
                    "createdAt"
                ),
                "updated_at": fulfillment.get(
                    "updatedAt"
                ),
                "delivered_at": fulfillment.get(
                    "deliveredAt"
                ),
                "estimated_delivery_at": (
                    fulfillment.get(
                        "estimatedDeliveryAt"
                    )
                ),
                "in_transit_at": fulfillment.get(
                    "inTransitAt"
                ),
                "tracking": tracking_items,
            }
        )

    return {
        "verified": True,
        "order_number": matched_order.get(
            "name"
        ),
        "order_status": matched_order.get(
            "displayFulfillmentStatus"
        ),
        "financial_status": matched_order.get(
            "displayFinancialStatus"
        ),
        "created_at": matched_order.get(
            "createdAt"
        ),
        "updated_at": matched_order.get(
            "updatedAt"
        ),
        "fulfillments": fulfillments,
    }
