from sqlalchemy import text

from database import SessionLocal


# ============================================================
# 1. SEARCH PRODUCTS
# ============================================================

def search_products(
    query: str = "",
    limit: int = 5,
    min_price: float | None = None,
    max_price: float | None = None,
    color: str | None = None,
    size: str | None = None,
    available_only: bool = False,
    category: str | None = None,
) -> list[dict]:
    """
    Search Naarira products using keywords and
    structured filters.
    """

    db = SessionLocal()

    try:

        conditions = []
        params = {
            "limit": limit
        }

        # ----------------------------------------------------
        # Keyword search
        # ----------------------------------------------------

        if query:

            conditions.append(
                """
                (
                    p.title ILIKE :query
                    OR p.description ILIKE :query
                    OR p.product_type ILIKE :query
                    OR p.vendor ILIKE :query
                    OR p.tags ILIKE :query
                    OR p.category ILIKE :query
                )
                """
            )

            params["query"] = f"%{query}%"

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        if category:

            conditions.append(
                """
                (
                    p.product_type ILIKE :category
                    OR p.category ILIKE :category
                    OR p.title ILIKE :category
                    OR p.description ILIKE :category
                    OR p.tags ILIKE :category
                )
                """
            )

            params["category"] = f"%{category}%"

        # ----------------------------------------------------
        # Minimum price
        # ----------------------------------------------------

        if min_price is not None:

            conditions.append(
                "CAST(v.price AS NUMERIC) >= :min_price"
            )

            params["min_price"] = min_price

        # ----------------------------------------------------
        # Maximum price
        # ----------------------------------------------------

        if max_price is not None:

            conditions.append(
                "CAST(v.price AS NUMERIC) <= :max_price"
            )

            params["max_price"] = max_price

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        if color:

            conditions.append(
                "v.color ILIKE :color"
            )

            params["color"] = f"%{color}%"

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        if size:

            conditions.append(
                "v.size ILIKE :size"
            )

            params["size"] = f"%{size}%"

        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        if available_only:

            conditions.append(
                "v.available = TRUE"
            )

        # ----------------------------------------------------
        # Build WHERE clause
        # ----------------------------------------------------

        where_clause = ""

        if conditions:

            where_clause = (
                "WHERE "
                + " AND ".join(conditions)
            )

        # ----------------------------------------------------
        # Execute query
        # ----------------------------------------------------

        sql = text(
            f"""
            SELECT DISTINCT
                p.id,
                p.shopify_product_id,
                p.title,
                p.handle,
                p.description,
                p.vendor,
                p.product_type,
                p.status,
                p.category,
                p.tags
            FROM products p
            JOIN product_variants v
                ON p.id = v.product_id

            {where_clause}

            ORDER BY p.id

            LIMIT :limit
            """
        )

        results = db.execute(
            sql,
            params
        ).fetchall()

        products = []

        for product in results:

            products.append({
                "product_id": product.id,
                "shopify_product_id": (
                    product.shopify_product_id
                ),
                "title": product.title,
                "handle": product.handle,
                "description": product.description,
                "vendor": product.vendor,
                "product_type": product.product_type,
                "status": product.status,
                "category": product.category,
                "tags": product.tags,
                "url": (
                    f"https://naarira.com/products/"
                    f"{product.handle}"
                )
            })

        return products

    finally:

        db.close()


# ============================================================
# 2. GET PRODUCT DETAILS
# ============================================================

def get_product_details(
    product_id: int
) -> dict | None:
    """
    Get complete product details including
    variants, prices, sizes, colors and availability.
    """

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # Product
        # ----------------------------------------------------

        product = db.execute(
            text("""
                SELECT
                    id,
                    shopify_product_id,
                    title,
                    handle,
                    description,
                    vendor,
                    product_type,
                    status,
                    category,
                    tags
                FROM products
                WHERE id = :product_id
            """),
            {
                "product_id": product_id
            }
        ).fetchone()

        if not product:

            return None

        # ----------------------------------------------------
        # Variants
        # ----------------------------------------------------

        variants = db.execute(
            text("""
                SELECT
                    id,
                    shopify_variant_id,
                    title,
                    sku,
                    price,
                    compare_at_price,
                    size,
                    color,
                    inventory_quantity,
                    available,
                    image_url
                FROM product_variants
                WHERE product_id = :product_id
                ORDER BY id
            """),
            {
                "product_id": product_id
            }
        ).fetchall()

        variant_list = []

        for variant in variants:

            variant_list.append({
                "variant_id": variant.id,
                "shopify_variant_id": (
                    variant.shopify_variant_id
                ),
                "title": variant.title,
                "sku": variant.sku,
                "price": (
                    float(variant.price)
                    if variant.price is not None
                    else None
                ),
                "compare_at_price": (
                    float(variant.compare_at_price)
                    if variant.compare_at_price is not None
                    else None
                ),
                "size": variant.size,
                "color": variant.color,
                "inventory_quantity": (
                    variant.inventory_quantity
                ),
                "available": variant.available,
                "image_url": variant.image_url
            })

        return {
            "product_id": product.id,
            "shopify_product_id": (
                product.shopify_product_id
            ),
            "title": product.title,
            "handle": product.handle,
            "description": product.description,
            "vendor": product.vendor,
            "product_type": product.product_type,
            "status": product.status,
            "category": product.category,
            "tags": product.tags,
            "url": (
                f"https://naarira.com/products/"
                f"{product.handle}"
            ),
            "variants": variant_list
        }

    finally:

        db.close()


# ============================================================
# 3. CHECK PRODUCT AVAILABILITY
# ============================================================

def check_product_availability(
    product_id: int
) -> dict | None:
    """
    Check product availability by variant,
    including size, color and inventory.
    """

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # Product
        # ----------------------------------------------------

        product = db.execute(
            text("""
                SELECT
                    id,
                    title,
                    handle
                FROM products
                WHERE id = :product_id
            """),
            {
                "product_id": product_id
            }
        ).fetchone()

        if not product:

            return None

        # ----------------------------------------------------
        # Variant availability
        # ----------------------------------------------------

        variants = db.execute(
            text("""
                SELECT
                    size,
                    color,
                    price,
                    inventory_quantity,
                    available
                FROM product_variants
                WHERE product_id = :product_id
                ORDER BY id
            """),
            {
                "product_id": product_id
            }
        ).fetchall()

        availability = []

        for variant in variants:

            availability.append({
                "size": variant.size,
                "color": variant.color,
                "price": (
                    float(variant.price)
                    if variant.price is not None
                    else None
                ),
                "inventory_quantity": (
                    variant.inventory_quantity
                ),
                "available": variant.available
            })

        return {
            "product_id": product.id,
            "title": product.title,
            "url": (
                f"https://naarira.com/products/"
                f"{product.handle}"
            ),
            "variants": availability
        }

    finally:

        db.close()
# ============================================================
# 4. TRACK ORDER
# ============================================================

def track_order(
    order_number: str,
    phone: str,
) -> dict:
    """
    Verify and track a Shopify order.
    """

    from shopify_client import (
        get_order_for_tracking,
    )

    order_number = str(
        order_number or ""
    ).strip()

    phone = str(
        phone or ""
    ).strip()

    if not order_number:

        return {
            "verified": False,
            "error": "Order number is required.",
        }

    if not phone:

        return {
            "verified": False,
            "error": "Phone number is required.",
        }

    try:

        order = get_order_for_tracking(
            order_number=order_number,
            phone=phone,
        )

    except Exception as exc:

        print(
            "Order tracking error:",
            type(exc).__name__,
            str(exc),
        )

        return {
            "verified": False,
            "error": (
                "Unable to retrieve the order "
                "right now. Please try again."
            ),
        }

    if not order:

        return {
            "verified": False,
            "error": (
                "We could not verify an order "
                "with that order number and phone number."
            ),
        }

    tracking_items = []

    for fulfillment in (
        order.get(
            "fulfillments"
        )
        or []
    ):

        for tracking in (
            fulfillment.get(
                "tracking"
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
                    "status": fulfillment.get(
                        "status"
                    ),
                    "delivered_at": fulfillment.get(
                        "delivered_at"
                    ),
                    "estimated_delivery_at": (
                        fulfillment.get(
                            "estimated_delivery_at"
                        )
                    ),
                }
            )

    order_status = (
        order.get(
            "order_status"
        )
        or "UNKNOWN"
    )

    order_number_value = (
        order.get(
            "order_number"
        )
        or order_number
    )

    if tracking_items:

        message = (
            f"Your order {order_number_value} "
            f"is currently "
            f"{order_status.lower()}."
        )

    else:

        message = (
            f"Your order {order_number_value} "
            f"is currently "
            f"{order_status.lower()}, "
            "but tracking information is not "
            "available yet."
        )

    return {
        "verified": True,
        "order_number": order_number_value,
        "status": order_status,
        "message": message,
        "tracking": tracking_items,
    }