from datetime import datetime

from database import Base, engine, SessionLocal
from models import Product, ProductVariant

from shopify_client import (
    get_access_token,
    fetch_all_products,
)


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


def sync_product(db, shopify_product):

    shopify_product_id = shopify_product["id"]

    product = (
        db.query(Product)
        .filter(
            Product.shopify_product_id
            == shopify_product_id
        )
        .first()
    )

    if not product:

        product = Product(
            shopify_product_id=shopify_product_id
        )

        db.add(product)

    product.title = shopify_product["title"]

    product.handle = shopify_product["handle"]

    product.description = shopify_product.get(
        "description"
    )

    product.vendor = shopify_product.get(
        "vendor"
    )

    product.product_type = shopify_product.get(
        "productType"
    )

    product.status = shopify_product.get(
        "status"
    )

    product.tags = ",".join(
        shopify_product.get("tags", [])
    )

    product.product_url = (
    f"https://naarira.com/products/"
    f"{shopify_product['handle']}"
)

    featured_image = shopify_product.get(
        "featuredImage"
    )

    if featured_image:

        product.image_url = featured_image.get(
            "url"
        )

    product.created_at = parse_shopify_datetime(
        shopify_product.get("createdAt")
    )

    product.updated_at = parse_shopify_datetime(
        shopify_product.get("updatedAt")
    )

    db.flush()

    return product


def sync_variant(
    db,
    product,
    shopify_variant
):

    shopify_variant_id = shopify_variant["id"]

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

    variant.title = shopify_variant.get(
        "title"
    )

    variant.sku = shopify_variant.get(
        "sku"
    )

    variant.price = shopify_variant.get(
        "price"
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

        variant.image_url = image.get(
            "url"
        )

    variant.created_at = parse_shopify_datetime(
        shopify_variant.get("createdAt")
    )

    variant.updated_at = parse_shopify_datetime(
        shopify_variant.get("updatedAt")
    )


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


def sync_products():

    print("=" * 60)

    print("NAARIRA SHOPIFY → POSTGRESQL SYNC")

    print("=" * 60)

    print()

    print("Creating database tables...")

    Base.metadata.create_all(
        bind=engine
    )

    print("Database ready.")

    print()

    print("Authenticating with Shopify...")

    access_token = get_access_token()

    print("Shopify authentication successful.")

    print()

    print("Fetching products...")

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
    variants_synced = 0

    try:

        for index, shopify_product in enumerate(
            products,
            start=1
        ):

            print(
                f"[{index}/{len(products)}] "
                f"{shopify_product['title']}"
            )

            existing = (
                db.query(Product)
                .filter(
                    Product.shopify_product_id
                    == shopify_product["id"]
                )
                .first()
            )

            if existing:

                products_updated += 1

            else:

                products_created += 1

            product = sync_product(
                db,
                shopify_product
            )

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

        db.commit()

        print()

        print("=" * 60)

        print("SYNC COMPLETED")

        print("=" * 60)

        print(
            f"Products fetched : {len(products)}"
        )

        print(
            f"Products created : {products_created}"
        )

        print(
            f"Products updated : {products_updated}"
        )

        print(
            f"Variants synced  : {variants_synced}"
        )

        print("=" * 60)

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


if __name__ == "__main__":

    sync_products()