"""
Naarira Phase A3
Policy / FAQ Semantic Search with Canonical Source Priority.

Features:
1. Gemini 3072-dimensional query embeddings
2. pgvector cosine similarity
3. Automatic policy category detection
4. Canonical source boosting
5. Similarity threshold
6. Clean structured results

Run from backend:

    python -m rag.knowledge_search
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy import text

from database import SessionLocal


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing in environment variables."
    )


# ============================================================
# GEMINI
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 3072

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SEARCH CONFIG
# ============================================================

DEFAULT_TOP_K = 5

MAX_TOP_K = 20

# We retrieve more candidates first and then rerank them.
CANDIDATE_LIMIT = 10

# Avoid returning extremely weak semantic matches.
MIN_SIMILARITY = 0.35


# ============================================================
# CANONICAL SOURCE PRIORITY
# ============================================================

CANONICAL_SOURCE_BOOST = {
    "shipping_policy.txt": 0.12,
    "return_refund_policy.txt": 0.12,
    "faqs.txt": 0.00,
}


# ============================================================
# QUERY CATEGORY DETECTION
# ============================================================

def detect_category(query: str) -> str | None:
    """
    Detect whether the user is asking about:

    - shipping
    - returns
    - FAQ/general

    This is intentionally lightweight and deterministic.
    """

    q = query.lower()

    # --------------------------------------------------------
    # RETURN / REFUND / EXCHANGE
    # --------------------------------------------------------

    return_keywords = [
        "return",
        "returns",
        "refund",
        "refunded",
        "money back",
        "exchange",
        "exchanges",
        "damaged",
        "defective",
        "wrong item",
        "incorrect item",
        "cancel",
        "cancellation",
        "non-returnable",
    ]

    if any(
        keyword in q
        for keyword in return_keywords
    ):
        return "returns"

    # --------------------------------------------------------
    # SHIPPING / DELIVERY
    # --------------------------------------------------------

    shipping_keywords = [
        "shipping",
        "ship",
        "delivery",
        "deliver",
        "free shipping",
        "shipping charge",
        "shipping cost",
        "how long",
        "tracking",
        "track my order",
        "delivery time",
        "delivery days",
        "dispatch",
        "dispatched",
        "address change",
    ]

    if any(
        keyword in q
        for keyword in shipping_keywords
    ):
        return "shipping"

    return None


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    query: str,
) -> list[float]:

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    if not response.embeddings:
        raise ValueError(
            "Gemini returned no embeddings."
        )

    embedding = response.embeddings[0].values

    if not embedding:
        raise ValueError(
            "Gemini returned an empty embedding."
        )

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"received {len(embedding)}."
        )

    return list(embedding)


# ============================================================
# PGVECTOR FORMAT
# ============================================================

def vector_literal(
    values: list[float],
) -> str:

    return (
        "["
        + ",".join(
            f"{value:.12f}"
            for value in values
        )
        + "]"
    )


# ============================================================
# CANONICAL BOOST
# ============================================================

def calculate_boost(
    source_file: str,
    detected_category: str | None,
) -> float:

    base_boost = CANONICAL_SOURCE_BOOST.get(
        source_file,
        0.0,
    )

    # --------------------------------------------------------
    # Category-specific preference
    # --------------------------------------------------------

    if (
        detected_category == "shipping"
        and source_file == "shipping_policy.txt"
    ):
        return base_boost + 0.10

    if (
        detected_category == "returns"
        and source_file == "return_refund_policy.txt"
    ):
        return base_boost + 0.10

    return base_boost


# ============================================================
# SEARCH
# ============================================================

def search_knowledge(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    category: str | None = None,
) -> list[dict[str, Any]]:

    query = query.strip()

    if not query:
        return []

    top_k = max(
        1,
        min(top_k, MAX_TOP_K),
    )

    detected_category = (
        category
        if category
        else detect_category(query)
    )

    print("\n" + "=" * 75)

    print(
        "NAARIRA POLICY / FAQ SEMANTIC SEARCH"
    )

    print("=" * 75)

    print(f"Query: {query}")

    print(
        f"Detected category: "
        f"{detected_category or 'general'}"
    )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    embedding = create_query_embedding(
        query
    )

    print(
        f"Query embedding dimension: "
        f"{len(embedding)}"
    )

    vector = vector_literal(
        embedding
    )

    # ========================================================
    # DATABASE SEARCH
    # ========================================================

    sql = text(
        """
        SELECT
            id,
            title,
            category,
            source_file,
            source_url,
            chunk_index,
            content,

            1 - (
                embedding <=> CAST(
                    :embedding AS vector
                )
            ) AS similarity

        FROM knowledge_documents

        WHERE embedding IS NOT NULL

        ORDER BY embedding <=> CAST(
            :embedding AS vector
        )

        LIMIT :candidate_limit
        """
    )

    with SessionLocal() as db:

        rows = (
            db.execute(
                sql,
                {
                    "embedding": vector,
                    "candidate_limit": CANDIDATE_LIMIT,
                },
            )
            .mappings()
            .all()
        )

    # ========================================================
    # RERANK
    # ========================================================

    reranked_results: list[dict[str, Any]] = []

    for row in rows:

        similarity = float(
            row["similarity"]
        )

        # Ignore extremely weak matches.
        if similarity < MIN_SIMILARITY:
            continue

        boost = calculate_boost(
            source_file=row["source_file"],
            detected_category=detected_category,
        )

        final_score = min(
            similarity + boost,
            1.0,
        )

        reranked_results.append(
            {
                "id": row["id"],
                "title": row["title"],
                "category": row["category"],
                "source_file": row["source_file"],
                "source_url": row["source_url"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "similarity": similarity,
                "boost": boost,
                "score": final_score,
            }
        )

    # ========================================================
    # SORT BY FINAL SCORE
    # ========================================================

    reranked_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    final_results = reranked_results[:top_k]

    # ========================================================
    # FINAL RANK
    # ========================================================

    for rank, result in enumerate(
        final_results,
        start=1,
    ):

        result["rank"] = rank

        print("\n" + "-" * 75)

        print(
            f"#{rank} {result['title']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Source: "
            f"{result['source_file']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print(
            f"Semantic similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Canonical boost: "
            f"+{result['boost']:.4f}"
        )

        print(
            f"Final score: "
            f"{result['score']:.4f}"
        )

        print(
            "\nContent:"
        )

        print(
            result["content"]
        )

    print("\n" + "=" * 75)

    print(
        f"Results returned: "
        f"{len(final_results)}"
    )

    print("=" * 75)

    return final_results


# ============================================================
# TEST QUERIES
# ============================================================

def main() -> None:

    test_queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "Do you offer free shipping?",
        "Do you offer exchanges?",
        "How long does a refund take?",
        "Can I change my shipping address?",
        "What happens if I receive a damaged item?",
    ]

    print("\n")

    print("#" * 75)

    print(
        "NAARIRA PHASE A3 — QUALITY TUNING TEST"
    )

    print("#" * 75)

    for query in test_queries:

        try:

            search_knowledge(
                query=query,
                top_k=3,
            )

        except Exception as exc:

            print("\n" + "!" * 75)

            print(
                f"Query failed: {query}"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: {exc}"
            )

            print("!" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()