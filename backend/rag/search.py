from sqlalchemy import text

from database import SessionLocal
from rag.embeddings import generate_embedding


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIMENSION = 3072

# Start conservatively because your real Naarira results
# were around 0.67 similarity.
MIN_SIMILARITY = 0.50


# ============================================================
# PGVECTOR FORMAT
# ============================================================

def embedding_to_pgvector(embedding: list[float]) -> str:
    """
    Convert a Python embedding list into PostgreSQL
    pgvector format.
    """

    return (
        "["
        + ",".join(
            str(value)
            for value in embedding
        )
        + "]"
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = 5,
    candidate_product_ids: list[int] | None = None,
):
    """
    Semantic search over Naarira product embeddings.

    Normal mode:

        query
            ↓
        all product embeddings
            ↓
        top_k semantic matches

    Hybrid mode:

        MCP filtered products
            ↓
        candidate_product_ids
            ↓
        semantic ranking only among candidates
    """

    print("\n" + "=" * 60)
    print("NAARIRA SEMANTIC SEARCH")
    print("=" * 60)

    print(f"\nQuery: {query}")

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = generate_embedding(
        query
    )

    dimension = len(
        query_embedding
    )

    print(
        f"Query embedding dimension: {dimension}"
    )

    if dimension != EMBEDDING_DIMENSION:

        raise ValueError(
            f"Unexpected query embedding dimension: "
            f"{dimension}. Expected {EMBEDDING_DIMENSION}."
        )

    # --------------------------------------------------------
    # Convert to pgvector format
    # --------------------------------------------------------

    embedding_string = embedding_to_pgvector(
        query_embedding
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        params = {
            "embedding": embedding_string,
            "top_k": top_k,
        }

        conditions = []

        # ----------------------------------------------------
        # Hybrid candidate filtering
        # ----------------------------------------------------

        if candidate_product_ids is not None:

            if not candidate_product_ids:

                print(
                    "\nNo candidate product IDs."
                )

                return []

            placeholders = []

            for index, product_id in enumerate(
                candidate_product_ids
            ):

                parameter_name = (
                    f"candidate_id_{index}"
                )

                placeholders.append(
                    f":{parameter_name}"
                )

                params[parameter_name] = int(
                    product_id
                )

            conditions.append(
                "pe.product_id IN ("
                + ", ".join(placeholders)
                + ")"
            )

            print(
                "\nHybrid retrieval enabled."
            )

            print(
                "Candidate products: "
                f"{len(candidate_product_ids)}"
            )

        # ----------------------------------------------------
        # WHERE clause
        # ----------------------------------------------------

        where_clause = ""

        if conditions:

            where_clause = (
                "WHERE "
                + " AND ".join(
                    conditions
                )
            )

        # ----------------------------------------------------
        # Vector similarity search
        # ----------------------------------------------------
        #
        # <=> = cosine distance in pgvector
        #
        # distance:
        #   smaller = more similar
        #
        # similarity:
        #   1 - distance
        # ----------------------------------------------------

        query_sql = text(
            f"""
            SELECT
                pe.product_id,
                p.title,
                p.handle,
                p.product_type,
                p.category,
                pe.content,

                1 - (
                    pe.embedding
                    <=> CAST(:embedding AS vector)
                ) AS similarity

            FROM product_embeddings pe

            INNER JOIN products p
                ON p.id = pe.product_id

            {where_clause}

            ORDER BY
                pe.embedding
                <=> CAST(:embedding AS vector)

            LIMIT :top_k
            """
        )

        results = db.execute(
            query_sql,
            params
        ).fetchall()

        # ----------------------------------------------------
        # Format results
        # ----------------------------------------------------

        formatted_results = []

        for result in results:

            similarity = float(
                result.similarity
            )

            # ------------------------------------------------
            # Similarity threshold
            # ------------------------------------------------

            if similarity < MIN_SIMILARITY:

                continue

            formatted_results.append({
                "rank": (
                    len(formatted_results) + 1
                ),
                "product_id": result.product_id,
                "title": result.title,
                "handle": result.handle,
                "product_type": result.product_type,
                "category": result.category,
                "similarity": similarity,
                "content": result.content,
            })

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "-" * 60)
        print("SEARCH RESULTS")
        print("-" * 60)

        if not formatted_results:

            print(
                "\nNo sufficiently relevant "
                "products found."
            )

            return []

        for product in formatted_results:

            print(
                f"\n#{product['rank']}"
            )

            print(
                f"Product ID : "
                f"{product['product_id']}"
            )

            print(
                f"Title      : "
                f"{product['title']}"
            )

            print(
                f"Category   : "
                f"{product.get('category') or product.get('product_type') or 'Not specified'}"
            )

            print(
                f"Similarity : "
                f"{product['similarity']:.4f}"
            )

            print(
                f"URL        : "
                f"https://naarira.com/products/"
                f"{product['handle']}"
            )

        print(
            f"\nRelevant results: "
            f"{len(formatted_results)}"
        )

        return formatted_results

    finally:

        db.close()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    user_query = input(
        "\nEnter search query: "
    ).strip()

    if not user_query:

        print(
            "Please enter a search query."
        )

    else:

        results = semantic_search(
            query=user_query,
            top_k=5
        )

        print(
            f"\nReturned {len(results)} results."
        )