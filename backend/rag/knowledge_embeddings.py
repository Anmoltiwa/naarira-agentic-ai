"""
Naarira Phase A2
Generate Gemini embeddings for knowledge_documents.

Run from backend:
    python -m rag.knowledge_embeddings
"""

from __future__ import annotations

import os
import time

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
# GEMINI CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 3072


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GENERATE EMBEDDING
# ============================================================

def embed_text(content: str) -> list[float]:
    """
    Generate a Gemini embedding for the supplied text.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
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

    # --------------------------------------------------------
    # Verify dimension
    # --------------------------------------------------------

    if len(embedding) != EMBEDDING_DIMENSION:

        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"received {len(embedding)}."
        )

    return list(embedding)


# ============================================================
# VECTOR CONVERSION
# ============================================================

def vector_literal(values: list[float]) -> str:
    """
    Convert Python list into pgvector literal format.
    """

    return (
        "["
        + ",".join(
            f"{value:.12f}"
            for value in values
        )
        + "]"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n")

    print("=" * 70)
    print("NAARIRA PHASE A2 — KNOWLEDGE EMBEDDINGS")
    print("=" * 70)

    print(
        f"\nEmbedding model:"
        f"\n{EMBEDDING_MODEL}"
    )

    print(
        f"\nEmbedding dimension:"
        f"\n{EMBEDDING_DIMENSION}"
    )

    # --------------------------------------------------------
    # Get documents without embeddings
    # --------------------------------------------------------

    with SessionLocal() as db:

        rows = (
            db.execute(
                text(
                    """
                    SELECT
                        id,
                        title,
                        category,
                        source_file,
                        chunk_index,
                        content
                    FROM knowledge_documents
                    WHERE embedding IS NULL
                    ORDER BY
                        source_file,
                        chunk_index
                    """
                )
            )
            .mappings()
            .all()
        )

    print(
        f"\nChunks requiring embeddings: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # Nothing to do
    # --------------------------------------------------------

    if not rows:

        print(
            "\n✓ All knowledge chunks already "
            "have embeddings."
        )

        return

    # --------------------------------------------------------
    # Process each chunk
    # --------------------------------------------------------

    success = 0
    failed = 0

    for index, row in enumerate(rows, start=1):

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(rows)}]"
        )

        print(
            f"ID: {row['id']}"
        )

        print(
            f"File: {row['source_file']}"
        )

        print(
            f"Category: {row['category']}"
        )

        print(
            f"Chunk: {row['chunk_index']}"
        )

        try:

            # ------------------------------------------------
            # Generate embedding
            # ------------------------------------------------

            embedding = embed_text(
                row["content"]
            )

            print(
                f"✓ Generated embedding "
                f"({len(embedding)} dimensions)"
            )

            # ------------------------------------------------
            # Convert to pgvector
            # ------------------------------------------------

            vector = vector_literal(
                embedding
            )

            # ------------------------------------------------
            # Save into database
            # ------------------------------------------------

            with SessionLocal() as db:

                db.execute(
                    text(
                        """
                        UPDATE knowledge_documents
                        SET
                            embedding =
                                CAST(
                                    :embedding
                                    AS vector
                                ),
                            updated_at =
                                CURRENT_TIMESTAMP
                        WHERE id = :id
                        """
                    ),
                    {
                        "embedding": vector,
                        "id": row["id"],
                    },
                )

                db.commit()

            success += 1

            print(
                "✓ Saved embedding to PostgreSQL"
            )

        except Exception as exc:

            failed += 1

            print(
                f"✗ Failed"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: {exc}"
            )

        # ----------------------------------------------------
        # Small delay to avoid hitting API too aggressively
        # ----------------------------------------------------

        time.sleep(0.5)

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n")

    print("=" * 70)

    print("EMBEDDING PROCESS COMPLETE")

    print("=" * 70)

    print(
        f"\nSuccessfully embedded: "
        f"{success}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    # --------------------------------------------------------
    # Database statistics
    # --------------------------------------------------------

    with SessionLocal() as db:

        stats = (
            db.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(embedding) AS embedded
                    FROM knowledge_documents
                    """
                )
            )
            .mappings()
            .one()
        )

        dimension = db.execute(
            text(
                """
                SELECT
                    vector_dims(embedding)
                FROM knowledge_documents
                WHERE embedding IS NOT NULL
                LIMIT 1
                """
            )
        ).scalar()

    print(
        f"\nKnowledge chunks in DB: "
        f"{stats['total']}"
    )

    print(
        f"Knowledge chunks with embeddings: "
        f"{stats['embedded']}"
    )

    print(
        f"Verified vector dimension: "
        f"{dimension}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if dimension == EMBEDDING_DIMENSION:

        print(
            "\n✓ Vector dimension verified: 3072"
        )

    else:

        print(
            "\n⚠ WARNING: Vector dimension "
            "does not match 3072!"
        )

    print("\n" + "=" * 70)

    print(
        "PHASE A2 EMBEDDINGS COMPLETE ✅"
    )

    print("=" * 70)

    print(
        "\nNext phase:"
    )

    print(
        "Phase A3 — Policy Semantic Search"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()