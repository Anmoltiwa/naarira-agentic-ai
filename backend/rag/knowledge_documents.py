"""
Naarira Phase A2
Ingest canonical policy / FAQ documents into PostgreSQL.

Run from backend:
    python -m rag.knowledge_documents
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

from sqlalchemy import text

from database import SessionLocal


# ============================================================
# PATHS
# ============================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR: Final[Path] = BASE_DIR / "knowledge"


# ============================================================
# KNOWLEDGE FILE CONFIGURATION
# ============================================================

FILE_CONFIG = {
    "shipping_policy.txt": {
        "title": "Naarira Shipping Policy",
        "category": "shipping",
        "source_url": "https://naarira.com/pages/shipping-policy",
    },
    "return_refund_policy.txt": {
        "title": "Naarira Return & Refund Policy",
        "category": "returns",
        "source_url": "https://naarira.com/pages/refund-policy",
    },
    "faqs.txt": {
        "title": "Naarira Frequently Asked Questions",
        "category": "faq",
        "source_url": "https://naarira.com/pages/faqs",
    },
}


# ============================================================
# DATABASE TABLE
# ============================================================

def ensure_table() -> None:
    """
    Make sure the knowledge_documents table exists.

    The pgvector extension is already installed because your
    existing product_embeddings table uses VECTOR(3072).
    """

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS knowledge_documents (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        source_file TEXT NOT NULL,
        source_url TEXT,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL UNIQUE,
        embedding VECTOR(3072),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_indexes_sql = [
        """
        CREATE INDEX IF NOT EXISTS knowledge_documents_category_idx
        ON knowledge_documents(category);
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_documents_source_file_idx
        ON knowledge_documents(source_file);
        """,
        """
        CREATE INDEX IF NOT EXISTS knowledge_documents_content_hash_idx
        ON knowledge_documents(content_hash);
        """,
    ]

    with SessionLocal() as db:

        db.execute(text(create_table_sql))

        for sql in create_indexes_sql:
            db.execute(text(sql))

        db.commit()

    print("✓ knowledge_documents table verified.")
    print("✓ Knowledge indexes verified.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value: str) -> str:
    """
    Normalize whitespace while keeping paragraphs.
    """

    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    value = re.sub(r"[ \t]+", " ", value)

    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_document(
    content: str,
    max_chars: int = 1200,
) -> list[str]:
    """
    Split knowledge document into chunks.

    First split on blank lines.
    Then merge small blocks.
    Large blocks are split line by line.
    """

    blocks = [
        normalize_text(block)
        for block in re.split(r"\n\s*\n", content)
        if normalize_text(block)
    ]

    chunks: list[str] = []

    current_blocks: list[str] = []
    current_length = 0

    for block in blocks:

        # ----------------------------------------------------
        # Small block
        # ----------------------------------------------------

        if len(block) <= max_chars:

            extra_length = 2 if current_blocks else 0

            proposed_length = (
                current_length
                + extra_length
                + len(block)
            )

            if proposed_length <= max_chars:

                current_blocks.append(block)

                current_length = proposed_length

                continue

            # Save accumulated chunk
            if current_blocks:

                chunks.append(
                    "\n\n".join(current_blocks)
                )

                current_blocks = []
                current_length = 0

            chunks.append(block)

            continue

        # ----------------------------------------------------
        # Large block
        # ----------------------------------------------------

        if current_blocks:

            chunks.append(
                "\n\n".join(current_blocks)
            )

            current_blocks = []
            current_length = 0

        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        partial = ""

        for line in lines:

            if not partial:

                partial = line

                continue

            proposed_length = (
                len(partial)
                + 1
                + len(line)
            )

            if proposed_length <= max_chars:

                partial = f"{partial} {line}"

            else:

                chunks.append(partial)

                partial = line

        if partial:

            chunks.append(partial)

    # --------------------------------------------------------
    # Final accumulated chunk
    # --------------------------------------------------------

    if current_blocks:

        chunks.append(
            "\n\n".join(current_blocks)
        )

    return chunks


# ============================================================
# CONTENT HASH
# ============================================================

def content_hash(content: str) -> str:
    """
    Generate SHA-256 hash for a chunk.
    """

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# INGEST ONE KNOWLEDGE FILE
# ============================================================

def ingest_file(
    filename: str,
    config: dict,
) -> int:

    file_path = KNOWLEDGE_DIR / filename

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nKnowledge file not found:\n"
            f"{file_path}\n\n"
            f"Expected file: {filename}"
        )

    print("\n" + "-" * 70)

    print(f"Processing: {filename}")

    print(f"Path: {file_path}")

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    raw_content = file_path.read_text(
        encoding="utf-8"
    )

    if not raw_content.strip():

        raise ValueError(
            f"{filename} is empty."
        )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = chunk_document(raw_content)

    print(
        f"Generated chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    with SessionLocal() as db:

        # ----------------------------------------------------
        # Remove previous version
        # ----------------------------------------------------

        db.execute(
            text(
                """
                DELETE FROM knowledge_documents
                WHERE source_file = :source_file
                """
            ),
            {
                "source_file": filename
            },
        )

        # ----------------------------------------------------
        # Insert new chunks
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            db.execute(
                text(
                    """
                    INSERT INTO knowledge_documents (
                        title,
                        category,
                        source_file,
                        source_url,
                        chunk_index,
                        content,
                        content_hash
                    )
                    VALUES (
                        :title,
                        :category,
                        :source_file,
                        :source_url,
                        :chunk_index,
                        :content,
                        :content_hash
                    )
                    """
                ),
                {
                    "title": config["title"],
                    "category": config["category"],
                    "source_file": filename,
                    "source_url": config["source_url"],
                    "chunk_index": index,
                    "content": chunk,
                    "content_hash": content_hash(chunk),
                },
            )

            print(
                f"  ✓ Chunk {index + 1} "
                f"({len(chunk)} chars)"
            )

        db.commit()

    print(
        f"✓ Successfully inserted "
        f"{len(chunks)} chunks"
    )

    return len(chunks)


# ============================================================
# VERIFY DATABASE
# ============================================================

def verify_database() -> None:

    with SessionLocal() as db:

        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_chunks,
                    COUNT(embedding) AS embedded_chunks
                FROM knowledge_documents
                """
            )
        ).mappings().one()

    print("\n" + "-" * 70)

    print(
        f"Total knowledge chunks: "
        f"{result['total_chunks']}"
    )

    print(
        f"Chunks with embeddings: "
        f"{result['embedded_chunks']}"
    )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n")

    print("=" * 70)

    print("NAARIRA PHASE A2")

    print("POLICY / FAQ KNOWLEDGE INGESTION")

    print("=" * 70)

    print(
        f"\nBackend directory:\n{BASE_DIR}"
    )

    print(
        f"\nKnowledge directory:\n{KNOWLEDGE_DIR}"
    )

    # --------------------------------------------------------
    # Check knowledge folder
    # --------------------------------------------------------

    if not KNOWLEDGE_DIR.exists():

        raise FileNotFoundError(
            f"\nKnowledge directory does not exist:\n"
            f"{KNOWLEDGE_DIR}\n\n"
            f"Create:\n"
            f"backend/knowledge/"
        )

    # --------------------------------------------------------
    # Verify expected files
    # --------------------------------------------------------

    for filename in FILE_CONFIG:

        path = KNOWLEDGE_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"\nMissing knowledge file:\n{path}"
            )

    # --------------------------------------------------------
    # Ensure table
    # --------------------------------------------------------

    print(
        "\nChecking knowledge_documents table..."
    )

    ensure_table()

    # --------------------------------------------------------
    # Ingest files
    # --------------------------------------------------------

    total_chunks = 0

    for filename, config in FILE_CONFIG.items():

        count = ingest_file(
            filename,
            config,
        )

        total_chunks += count

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    verify_database()

    print("\n" + "=" * 70)

    print("PHASE A2 INGESTION COMPLETE ✅")

    print("=" * 70)

    print(
        f"\nTotal chunks inserted: {total_chunks}"
    )

    print(
        "\nNext command:"
    )

    print(
        "python -m rag.knowledge_embeddings"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()