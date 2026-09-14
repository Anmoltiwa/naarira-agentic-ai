from sqlalchemy.orm import Session

from database import SessionLocal
from models import Product


def create_product_document(product):
    """
    Convert a Shopify product and its variants
    into a clean text document for RAG.
    """

    lines = []

    lines.append(f"Product: {product.title}")

    if product.product_type:
        lines.append(f"Category: {product.product_type}")

    if product.vendor:
        lines.append(f"Brand: {product.vendor}")

    if product.description:
        lines.append(f"Description: {product.description}")

    if product.tags:
        lines.append(f"Tags: {product.tags}")

    if product.status:
        lines.append(f"Status: {product.status}")

    if product.handle:
        lines.append(
            f"Product URL: https://naarira.com/products/{product.handle}"
        )

    # Collect variant information
    sizes = set()
    colors = set()
    prices = []
    availability = []

    for variant in product.variants:

        if variant.size:
            sizes.add(variant.size)

        if variant.color:
            colors.add(variant.color)

        if variant.price:
            prices.append(str(variant.price))

        if variant.available:
            availability.append("In Stock")

    if sizes:
        lines.append(
            f"Available Sizes: {', '.join(sorted(sizes))}"
        )

    if colors:
        lines.append(
            f"Available Colors: {', '.join(sorted(colors))}"
        )

    if prices:
        unique_prices = sorted(set(prices))
        lines.append(
            f"Prices: ₹{', ₹'.join(unique_prices)}"
        )

    if availability:
        lines.append("Availability: In Stock")
    else:
        lines.append("Availability: Currently unavailable")

    return "\n".join(lines)


def get_all_product_documents():
    """
    Read products from PostgreSQL and convert them
    into RAG documents.
    """

    db: Session = SessionLocal()

    try:
        products = db.query(Product).all()

        documents = []

        for product in products:

            content = create_product_document(product)

            documents.append({
                "product_id": product.id,
                "shopify_product_id": product.shopify_product_id,
                "content": content
            })

        return documents

    finally:
        db.close()


if __name__ == "__main__":

    print("=" * 60)
    print("NAARIRA RAG DOCUMENT GENERATION")
    print("=" * 60)

    documents = get_all_product_documents()

    print(f"\nProducts found: {len(documents)}")

    for document in documents[:5]:

        print("\n" + "-" * 60)
        print(f"Product ID: {document['product_id']}")
        print(document["content"])

    print("\n" + "=" * 60)
    print("RAG DOCUMENT GENERATION COMPLETE")
    print("=" * 60)