import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sqlalchemy import text

from database import SessionLocal
from rag.documents import get_all_product_documents


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


# ---------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"


# ---------------------------------------------------------
# Generate embedding
# ---------------------------------------------------------

def generate_embedding(text_content: str):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text_content
    )

    embedding = response.embeddings[0].values

    return embedding


# ---------------------------------------------------------
# Convert embedding to PostgreSQL vector format
# ---------------------------------------------------------

def vector_to_pgvector(embedding):

    return "[" + ",".join(
        str(value) for value in embedding
    ) + "]"


# ---------------------------------------------------------
# Store embeddings in PostgreSQL
# ---------------------------------------------------------

def store_product_embeddings():

    print("\n" + "=" * 60)
    print("NAARIRA PRODUCT EMBEDDING GENERATION")
    print("=" * 60)

    documents = get_all_product_documents()

    print(f"\nProducts found: {len(documents)}")

    if not documents:
        print("No products found.")
        return

    db = SessionLocal()

    processed = 0
    skipped = 0
    failed = 0

    try:

        for document in documents:

            product_id = document["product_id"]
            content = document["content"]

            print("\n" + "-" * 60)
            print(f"Product ID: {product_id}")
            print(f"Product: {content.splitlines()[0]}")

            # -------------------------------------------------
            # Check existing embedding BEFORE calling Gemini
            # -------------------------------------------------

            existing = db.execute(
                text("""
                    SELECT id
                    FROM product_embeddings
                    WHERE product_id = :product_id
                """),
                {
                    "product_id": product_id
                }
            ).fetchone()

            if existing:

                print(
                    "Embedding already exists."
                    " Skipping Gemini API."
                )

                skipped += 1
                continue

            # -------------------------------------------------
            # Generate embedding
            # -------------------------------------------------

            print("Generating new Gemini embedding...")

            try:

                embedding = generate_embedding(content)

            except errors.ClientError as e:

                error_message = str(e)

                print("\nGEMINI API ERROR")
                print(error_message)

                # ---------------------------------------------
                # Handle quota exhaustion
                # ---------------------------------------------

                if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

                    print("\n" + "=" * 60)
                    print("GEMINI QUOTA EXHAUSTED")
                    print("=" * 60)

                    print(
                        "\nEmbedding generation stopped safely."
                    )

                    print(
                        f"Last product not embedded: "
                        f"{product_id}"
                    )

                    print(
                        "\nExisting embeddings are safe."
                    )

                    print(
                        "Run this script again when "
                        "Gemini quota becomes available."
                    )

                    break

                # ---------------------------------------------
                # Other Gemini errors
                # ---------------------------------------------

                failed += 1

                print(
                    "Gemini error. Skipping this product."
                )

                continue

            # -------------------------------------------------
            # Verify dimension
            # -------------------------------------------------

            dimension = len(embedding)

            print(
                f"Embedding dimension: {dimension}"
            )

            if dimension != 3072:

                raise ValueError(
                    f"Unexpected embedding dimension: "
                    f"{dimension}. Expected 3072."
                )

            # -------------------------------------------------
            # Convert to pgvector format
            # -------------------------------------------------

            embedding_string = vector_to_pgvector(
                embedding
            )

            # -------------------------------------------------
            # Insert embedding
            # -------------------------------------------------

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
                    "embedding": embedding_string
                }
            )

            db.commit()

            processed += 1

            print(
                f"Stored successfully "
                f"({processed} new embedding)"
            )

            # -------------------------------------------------
            # Small delay
            # -------------------------------------------------

            time.sleep(0.2)

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("EMBEDDING PROCESS SUMMARY")
        print("=" * 60)

        print(
            f"\nNew embeddings created : {processed}"
        )

        print(
            f"Existing embeddings    : {skipped}"
        )

        print(
            f"Failed products        : {failed}"
        )

        # -----------------------------------------------------
        # Current database count
        # -----------------------------------------------------

        result = db.execute(
            text("""
                SELECT COUNT(*)
                FROM product_embeddings
            """)
        ).scalar()

        print(
            f"Total embeddings in DB : {result}"
        )

        print("\n" + "=" * 60)

    except Exception as e:

        db.rollback()

        print("\nERROR:")
        print(str(e))

        raise

    finally:

        db.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("NAARIRA RAG - GEMINI EMBEDDINGS")
    print("=" * 60)

    store_product_embeddings()