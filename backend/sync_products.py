from datetime import datetime

from database import Base, engine, SessionLocal
from models import Product, ProductVariant

from shopify_client import (
    get_access_token,
    fetch_all_products,
)

from sqlalchemy import text


# =========================================================
# OPTION HELPERS
# =========================================================

def extract_option(variant, option_name):

    for option in variant.get("selectedOptions", []):

        if option["name"].lower() == option_name.lower():

            return option["value"]

    return None


def extract_variant_options(variant):

    size = extract_option(
        variant,
        "Size"
    )

    color = extract_option(
        variant,
        "Color"
    )

    return size, color


# =========================================================
# DATETIME
# =========================================================

def parse_shopify_datetime(value):

    if not value:
        return None

    try:

        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00"
            )
        )

    except ValueError:

        return None


# =========================================================
# PRODUCT SYNC
# =========================================================

def sync_product(
    db,
    shopify_product
):

    shopify_product_id = shopify_product["id"]

    product = (
        db.query(Product)
        .filter(
            Product.shopify_product_id
            == shopify_product_id
        )
        .first()
    )

    is_new = False

    if not product:

        product = Product(
            shopify_product_id=shopify_product_id
        )

        db.add(product)

        is_new = True

    # -----------------------------------------------------
    # Store Shopify values
    # -----------------------------------------------------

    product.title = shopify_product["title"]

    product.handle = shopify_product["handle"]

    product.description = (
        shopify_product.get("description")
    )

    product.vendor = (
        shopify_product.get("vendor")
    )

    product.product_type = (
        shopify_product.get("productType")
    )

    product.status = (
        shopify_product.get("status")
    )

    product.tags = ",".join(
        shopify_product.get("tags", [])
    )

    product.product_url = (
        f"https://naarira.com/products/"
        f"{shopify_product['handle']}"
    )

    featured_image = (
        shopify_product.get("featuredImage")
    )

    if featured_image:

        product.image_url = (
            featured_image.get("url")
        )

    product.created_at = (
        parse_shopify_datetime(
            shopify_product.get("createdAt")
        )
    )

    product.updated_at = (
        parse_shopify_datetime(
            shopify_product.get("updatedAt")
        )
    )

    db.flush()

    return product, is_new


# =========================================================
# VARIANT SYNC
# =========================================================

def sync_variant(
    db,
    product,
    shopify_variant
):

    shopify_variant_id = (
        shopify_variant["id"]
    )

    variant = (
        db.query(ProductVariant)
        .filter(
            ProductVariant.shopify_variant_id
            == shopify_variant_id
        )
        .first()
    )

    if not variant:

        variant = ProductVariant(
            shopify_variant_id=shopify_variant_id,
            product_id=product.id
        )

        db.add(variant)

    variant.product_id = product.id

    variant.title = (
        shopify_variant.get("title")
    )

    variant.sku = (
        shopify_variant.get("sku")
    )

    variant.price = (
        shopify_variant.get("price")
    )

    variant.compare_at_price = (
        shopify_variant.get(
            "compareAtPrice"
        )
    )

    size, color = extract_variant_options(
        shopify_variant
    )

    variant.size = size

    variant.color = color

    variant.inventory_quantity = (
        shopify_variant.get(
            "inventoryQuantity"
        )
    )

    variant.available = (
        shopify_variant.get(
            "availableForSale",
            False
        )
    )

    image = shopify_variant.get("image")

    if image:

        variant.image_url = (
            image.get("url")
        )

    variant.created_at = (
        parse_shopify_datetime(
            shopify_variant.get("createdAt")
        )
    )

    variant.updated_at = (
        parse_shopify_datetime(
            shopify_variant.get("updatedAt")
        )
    )


# =========================================================
# CHECK WHETHER PRODUCT CHANGED
# =========================================================

def product_has_changed(
    existing_product,
    shopify_product
):

    if not existing_product:

        return True

    shopify_updated_at = (
        parse_shopify_datetime(
            shopify_product.get("updatedAt")
        )
    )

    local_updated_at = (
        existing_product.updated_at
    )

    # If either timestamp is missing,
    # treat product as changed.
    if not shopify_updated_at:
        return True

    if not local_updated_at:
        return True

    # PostgreSQL datetime may be timezone-naive.
    # Normalize it to the Shopify timestamp timezone.
    if (
        local_updated_at.tzinfo is None
        and shopify_updated_at.tzinfo is not None
    ):

        local_updated_at = (
            local_updated_at.replace(
                tzinfo=shopify_updated_at.tzinfo
            )
        )

    return (
        shopify_updated_at
        > local_updated_at
    )


# =========================================================
# MAIN SYNC
# =========================================================

def sync_products():

    print("=" * 60)

    print(
        "NAARIRA SHOPIFY → POSTGRESQL SYNC"
    )

    print("=" * 60)

    print()

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    print(
        "Creating database tables..."
    )

    Base.metadata.create_all(
        bind=engine
    )

    print(
        "Database ready."
    )

    print()

    # -----------------------------------------------------
    # Shopify authentication
    # -----------------------------------------------------

    print(
        "Authenticating with Shopify..."
    )

    access_token = get_access_token()

    print(
        "Shopify authentication successful."
    )

    print()

    # -----------------------------------------------------
    # Fetch products
    # -----------------------------------------------------

    print(
        "Fetching products..."
    )

    products = fetch_all_products(
        access_token
    )

    print()

    print(
        f"Total Shopify products fetched: "
        f"{len(products)}"
    )

    print()

    db = SessionLocal()

    products_created = 0
    products_updated = 0
    products_unchanged = 0
    variants_synced = 0

    changed_product_ids = []

    try:

        # =================================================
        # PROCESS PRODUCTS
        # =================================================

        for index, shopify_product in enumerate(
            products,
            start=1
        ):

            print(
                f"[{index}/{len(products)}] "
                f"{shopify_product['title']}"
            )

            shopify_product_id = (
                shopify_product["id"]
            )

            # -------------------------------------------------
            # Find existing local product BEFORE updating it
            # -------------------------------------------------

            existing_product = (
                db.query(Product)
                .filter(
                    Product.shopify_product_id
                    == shopify_product_id
                )
                .first()
            )

            # -------------------------------------------------
            # Determine change
            # -------------------------------------------------

            changed = product_has_changed(
                existing_product,
                shopify_product
            )

            if existing_product is None:

                print(
                    "  → NEW PRODUCT"
                )

                products_created += 1

            elif changed:

                print(
                    "  → PRODUCT CHANGED"
                )

                products_updated += 1

            else:

                print(
                    "  → PRODUCT UNCHANGED"
                )

                products_unchanged += 1

            # -------------------------------------------------
            # Sync product
            # -------------------------------------------------

            product, is_new = sync_product(
                db,
                shopify_product
            )

            # -------------------------------------------------
            # Sync variants
            # -------------------------------------------------

            variant_edges = (
                shopify_product
                .get("variants", {})
                .get("edges", [])
            )

            for edge in variant_edges:

                variant = edge["node"]

                sync_variant(
                    db,
                    product,
                    variant
                )

                variants_synced += 1

            # -------------------------------------------------
            # Remember products needing embeddings
            # -------------------------------------------------

            if changed:

                changed_product_ids.append(
                    product.id
                )

                print(
                    f"  → Added to embedding queue: "
                    f"{product.id}"
                )

        # =================================================
        # COMMIT PRODUCT SYNC
        # =================================================

        db.commit()

        # =================================================
        # SUMMARY
        # =================================================

        print()

        print("=" * 60)

        print(
            "SHOPIFY SYNC COMPLETED"
        )

        print("=" * 60)

        print(
            f"Products fetched     : "
            f"{len(products)}"
        )

        print(
            f"Products created     : "
            f"{products_created}"
        )

        print(
            f"Products changed     : "
            f"{products_updated}"
        )

        print(
            f"Products unchanged   : "
            f"{products_unchanged}"
        )

        print(
            f"Variants synced      : "
            f"{variants_synced}"
        )

        print(
            f"Embedding candidates : "
            f"{len(changed_product_ids)}"
        )

        print("=" * 60)

        # =================================================
        # IMPORTANT
        # =================================================

        print()

        print(
            "Changed product IDs:"
        )

        print(
            changed_product_ids
        )

        print()

        print(
            "Shopify → PostgreSQL sync finished."
        )

        print(
            "Embedding generation will be handled separately."
        )

        print("=" * 60)

        return changed_product_ids

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    sync_products()