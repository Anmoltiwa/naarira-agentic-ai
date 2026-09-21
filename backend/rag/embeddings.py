import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sqlalchemy import text

from database import SessionLocal
from rag.documents import get_all_product_documents


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-001"

EXPECTED_DIMENSION = 3072


# =========================================================
# GENERATE EMBEDDING
# =========================================================

def generate_embedding(text_content: str):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text_content,
    )

    embedding = response.embeddings[0].values

    return embedding


# =========================================================
# CONVERT TO PGVECTOR
# =========================================================

def vector_to_pgvector(embedding):

    return "[" + ",".join(
        str(value)
        for value in embedding
    ) + "]"


# =========================================================
# GET EXISTING EMBEDDING
# =========================================================

def get_existing_embedding(
    db,
    product_id,
):

    return db.execute(
        text("""
            SELECT
                id,
                content
            FROM product_embeddings
            WHERE product_id = :product_id
        """),
        {
            "product_id": product_id
        },
    ).fetchone()


# =========================================================
# INSERT NEW EMBEDDING
# =========================================================

def insert_embedding(
    db,
    product_id,
    content,
    embedding_string,
):

    db.execute(
        text("""
            INSERT INTO product_embeddings
            (
                product_id,
                content,
                embedding
            )
            VALUES
            (
                :product_id,
                :content,
                CAST(:embedding AS vector)
            )
        """),
        {
            "product_id": product_id,
            "content": content,
            "embedding": embedding_string,
        },
    )


# =========================================================
# UPDATE EXISTING EMBEDDING
# =========================================================

def update_embedding(
    db,
    embedding_id,
    content,
    embedding_string,
):

    db.execute(
        text("""
            UPDATE product_embeddings
            SET
                content = :content,
                embedding = CAST(:embedding AS vector),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :embedding_id
        """),
        {
            "embedding_id": embedding_id,
            "content": content,
            "embedding": embedding_string,
        },
    )


# =========================================================
# MAIN EMBEDDING SYNC
# =========================================================

def store_product_embeddings(
    candidate_product_ids=None,
):

    print("\n" + "=" * 60)
    print("NAARIRA PRODUCT EMBEDDING SYNC")
    print("=" * 60)

    # -----------------------------------------------------
    # Get all product documents
    # -----------------------------------------------------

    documents = get_all_product_documents()

    print(
        f"\nTotal product documents found: "
        f"{len(documents)}"
    )

    if not documents:

        print("No products found.")

        return

    # -----------------------------------------------------
    # Optional candidate filtering
    # -----------------------------------------------------

    if candidate_product_ids is not None:

        candidate_product_ids = set(
            candidate_product_ids
        )

        documents = [
            document
            for document in documents
            if document["product_id"]
            in candidate_product_ids
        ]

        print(
            f"Embedding candidates: "
            f"{len(documents)}"
        )

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    db = SessionLocal()

    created = 0
    updated = 0
    unchanged = 0
    failed = 0

    try:

        for document in documents:

            product_id = document["product_id"]

            content = document["content"]

            print("\n" + "-" * 60)

            print(
                f"Product ID: {product_id}"
            )

            print(
                f"Product: "
                f"{content.splitlines()[0]}"
            )

            # -------------------------------------------------
            # Check existing embedding
            # -------------------------------------------------

            existing = get_existing_embedding(
                db,
                product_id,
            )

            # =================================================
            # EXISTING EMBEDDING
            # =================================================

            if existing:

                embedding_id = existing[0]

                existing_content = existing[1]

                # ---------------------------------------------
                # Product has NOT changed
                # ---------------------------------------------

                if existing_content == content:

                    print(
                        "Embedding unchanged."
                    )

                    print(
                        "Skipping Gemini API."
                    )

                    unchanged += 1

                    continue

                # ---------------------------------------------
                # Product content changed
                # ---------------------------------------------

                print(
                    "Product content changed."
                )

                print(
                    "Generating new embedding..."
                )

                operation = "update"

            # =================================================
            # NEW EMBEDDING
            # =================================================

            else:

                print(
                    "No embedding found."
                )

                print(
                    "Generating new embedding..."
                )

                operation = "insert"

                embedding_id = None

            # -------------------------------------------------
            # Gemini
            # -------------------------------------------------

            try:

                embedding = generate_embedding(
                    content
                )

            except errors.ClientError as e:

                error_message = str(e)

                print("\nGEMINI API ERROR")

                print(error_message)

                # ---------------------------------------------
                # Quota exhausted
                # ---------------------------------------------

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED"
                    in error_message
                ):

                    print(
                        "\n"
                        + "=" * 60
                    )

                    print(
                        "GEMINI QUOTA EXHAUSTED"
                    )

                    print(
                        "=" * 60
                    )

                    print(
                        "Stopping embedding sync safely."
                    )

                    print(
                        f"Product not processed: "
                        f"{product_id}"
                    )

                    print(
                        "Already committed embeddings "
                        "are safe."
                    )

                    break

                # ---------------------------------------------
                # Other Gemini error
                # ---------------------------------------------

                failed += 1

                print(
                    "Gemini error."
                )

                print(
                    "Skipping this product."
                )

                continue

            # -------------------------------------------------
            # Verify dimension
            # -------------------------------------------------

            dimension = len(embedding)

            print(
                f"Embedding dimension: "
                f"{dimension}"
            )

            if dimension != EXPECTED_DIMENSION:

                raise ValueError(
                    "Unexpected embedding dimension: "
                    f"{dimension}. "
                    f"Expected "
                    f"{EXPECTED_DIMENSION}."
                )

            # -------------------------------------------------
            # Convert to pgvector
            # -------------------------------------------------

            embedding_string = (
                vector_to_pgvector(
                    embedding
                )
            )

            # =================================================
            # UPDATE
            # =================================================

            if operation == "update":

                update_embedding(
                    db=db,
                    embedding_id=embedding_id,
                    content=content,
                    embedding_string=embedding_string,
                )

                db.commit()

                updated += 1

                print(
                    "✓ Embedding updated successfully."
                )

            # =================================================
            # INSERT
            # =================================================

            else:

                insert_embedding(
                    db=db,
                    product_id=product_id,
                    content=content,
                    embedding_string=embedding_string,
                )

                db.commit()

                created += 1

                print(
                    "✓ Embedding created successfully."
                )

            # -------------------------------------------------
            # Small delay
            # -------------------------------------------------

            time.sleep(0.2)

        # =====================================================
        # FINAL SUMMARY
        # =====================================================

        print("\n" + "=" * 60)

        print(
            "EMBEDDING SYNC SUMMARY"
        )

        print("=" * 60)

        print(
            f"New embeddings created : "
            f"{created}"
        )

        print(
            f"Embeddings updated     : "
            f"{updated}"
        )

        print(
            f"Embeddings unchanged   : "
            f"{unchanged}"
        )

        print(
            f"Failed products        : "
            f"{failed}"
        )

        # -----------------------------------------------------
        # Current DB count
        # -----------------------------------------------------

        total = db.execute(
            text("""
                SELECT COUNT(*)
                FROM product_embeddings
            """)
        ).scalar()

        print(
            f"Total embeddings in DB : "
            f"{total}"
        )

        print("=" * 60)

    except Exception as e:

        db.rollback()

        print("\nERROR:")

        print(str(e))

        raise

    finally:

        db.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "NAARIRA RAG - PRODUCT EMBEDDING SYNC"
    )

    print("=" * 60)

    store_product_embeddings()