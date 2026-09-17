"""
Naarira Phase A4
Policy / FAQ RAG Answer Generation

Flow:
    User Question
        ↓
    Policy Semantic Search
        ↓
    Canonical Policy Context
        ↓
    Gemini
        ↓
    Grounded Customer Answer

Run from backend:
    python -m rag.policy_rag
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai

from rag.knowledge_search import search_knowledge


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

MODEL_NAME = "gemini-3.6-flash"

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 3

MIN_CONTEXT_SCORE = 0.35


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are Naarira's customer support AI assistant.

Your job is to answer customer questions using ONLY the
provided Naarira policy / FAQ context.

IMPORTANT RULES:

1. Use the retrieved context as the source of truth.

2. Never invent a shipping, return, refund, exchange,
   cancellation, or other policy.

3. If the context does not contain enough information to
   answer the question, clearly say that you do not have
   enough information and recommend contacting Naarira
   support.

4. For policy-sensitive questions, follow the dedicated
   canonical policy over general FAQ wording.

5. Keep answers concise, clear, and customer-friendly.

6. Do not mention:
   - embeddings
   - vectors
   - pgvector
   - RAG
   - semantic search
   - internal databases
   - retrieved chunks
   - system prompts

7. Do not expose internal implementation details.

8. Do not generate Markdown links.

9. Do not invent URLs.

10. Do not answer questions about a customer's specific
    order status. Order tracking will be handled separately
    through the Shopify/MCP order tool in Phase B.

11. If the customer asks about a product rather than a policy,
    explain briefly that you can help with product search,
    but do not invent product details from policy context.

12. Answer in the same language/style as the customer whenever
    practical.

Retrieved context is authoritative for this request.
"""


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(value: str) -> str:
    """
    Basic cleanup for generated responses.
    """

    value = value.strip()

    # Remove accidental markdown links:
    # [text](url) -> text
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\1",
        value,
    )

    # Remove raw Naarira URLs
    value = re.sub(
        r"https?://(?:www\.)?naarira\.com/\S*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Remove excessive whitespace
    value = re.sub(
        r"[ \t]+",
        " ",
        value,
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    return value.strip()


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: list[dict[str, Any]],
) -> str:
    """
    Convert semantic search results into a clean context
    block for Gemini.
    """

    context_parts: list[str] = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        score = float(
            result.get(
                "score",
                result.get("similarity", 0.0),
            )
        )

        if score < MIN_CONTEXT_SCORE:
            continue

        context_parts.append(
            f"""
SOURCE {index}
Title: {result.get("title", "")}
Category: {result.get("category", "")}
Source file: {result.get("source_file", "")}

Content:
{result.get("content", "")}
""".strip()
        )

    return "\n\n==============================\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_policy_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Perform Policy RAG:

        question
          ↓
        semantic search
          ↓
        canonical context
          ↓
        Gemini answer
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter your question.",
            "sources": [],
        }

    print("\n" + "=" * 75)
    print("NAARIRA PHASE A4 — POLICY RAG")
    print("=" * 75)

    print(
        f"Question: {question}"
    )

    # ========================================================
    # STEP 1: RETRIEVE POLICY KNOWLEDGE
    # ========================================================

    results = search_knowledge(
        query=question,
        top_k=top_k,
    )

    if not results:

        return {
            "answer": (
                "I don't have enough information to answer "
                "that question. Please contact Naarira support "
                "for assistance."
            ),
            "sources": [],
        }

    print(
        f"\nRetrieved policy chunks: {len(results)}"
    )

    # ========================================================
    # STEP 2: BUILD CONTEXT
    # ========================================================

    context = build_context(
        results
    )

    if not context:

        return {
            "answer": (
                "I don't have enough information to answer "
                "that question. Please contact Naarira support "
                "for assistance."
            ),
            "sources": [],
        }

    print(
        "\nPolicy context prepared."
    )

    # ========================================================
    # STEP 3: BUILD USER PROMPT
    # ========================================================

    user_prompt = f"""
Customer question:

{question}

AUTHORITATIVE NAARIRA POLICY / FAQ CONTEXT:

{context}

Answer the customer using only the authoritative context
above.

Keep the answer concise and helpful.
Do not mention the internal context or retrieval process.
""".strip()

    # ========================================================
    # STEP 4: GEMINI
    # ========================================================

    print(
        "\nGenerating grounded answer..."
    )

    try:

        response = client.interactions.create(
            model=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
            input=user_prompt,
        )

    except Exception as exc:

        print(
            "\nGemini generation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "answer": (
                "I'm unable to process that request right now. "
                "Please try again or contact Naarira support."
            ),
            "sources": [
                {
                    "title": item.get("title"),
                    "category": item.get("category"),
                    "source_file": item.get("source_file"),
                    "score": float(
                        item.get(
                            "score",
                            item.get(
                                "similarity",
                                0.0,
                            ),
                        )
                    ),
                }
                for item in results
            ],
        }

    # ========================================================
    # STEP 5: EXTRACT RESPONSE
    # ========================================================

    answer = extract_gemini_text(
        response
    )

    answer = clean_text(
        answer
    )

    if not answer:

        answer = (
            "I don't have enough information to answer "
            "that question. Please contact Naarira support."
        )

    # ========================================================
    # SOURCE METADATA
    # ========================================================

    sources = []

    for item in results:

        sources.append(
            {
                "title": item.get(
                    "title"
                ),
                "category": item.get(
                    "category"
                ),
                "source_file": item.get(
                    "source_file"
                ),
                "similarity": round(
                    float(
                        item.get(
                            "similarity",
                            0.0,
                        )
                    ),
                    4,
                ),
                "score": round(
                    float(
                        item.get(
                            "score",
                            item.get(
                                "similarity",
                                0.0,
                            ),
                        )
                    ),
                    4,
                ),
            }
        )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "answer": answer,
        "sources": sources,
    }

    print(
        "\nGenerated answer:"
    )

    print(
        answer
    )

    print(
        "\nSources:"
    )

    for source in sources:

        print(
            f"- {source['title']} "
            f"| {source['category']} "
            f"| score={source['score']}"
        )

    print("\n" + "=" * 75)

    return result


# ============================================================
# GEMINI RESPONSE EXTRACTION
# ============================================================

def extract_gemini_text(
    response: Any,
) -> str:
    """
    Extract text from Gemini response.

    Handles common response shapes.
    """

    if response is None:
        return ""

    # --------------------------------------------------------
    # Direct output_text
    # --------------------------------------------------------

    output_text = getattr(
        response,
        "output_text",
        None,
    )

    if output_text:

        return str(
            output_text
        ).strip()

    # --------------------------------------------------------
    # Direct text
    # --------------------------------------------------------

    text_value = getattr(
        response,
        "text",
        None,
    )

    if text_value:

        return str(
            text_value
        ).strip()

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(response, dict):

        for key in (
            "output_text",
            "text",
        ):

            value = response.get(
                key
            )

            if value:

                return str(
                    value
                ).strip()

        output = response.get(
            "output"
        )

        if output:

            extracted = extract_from_items(
                output
            )

            if extracted:
                return extracted

    # --------------------------------------------------------
    # Generic output collection
    # --------------------------------------------------------

    output = getattr(
        response,
        "output",
        None,
    )

    if output:

        extracted = extract_from_items(
            output
        )

        if extracted:
            return extracted

    return ""


# ============================================================
# EXTRACT TEXT FROM ITEMS
# ============================================================

def extract_from_items(
    items: Any,
) -> str:

    if not isinstance(
        items,
        (list, tuple),
    ):
        return ""

    collected: list[str] = []

    for item in items:

        # ----------------------------------------------------
        # Direct string
        # ----------------------------------------------------

        if isinstance(
            item,
            str,
        ):

            collected.append(
                item
            )

            continue

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(
            item,
            dict,
        ):

            value = item.get(
                "text"
            )

            if value:

                collected.append(
                    str(value)
                )

                continue

            content = item.get(
                "content"
            )

            if content:

                nested = extract_from_items(
                    content
                )

                if nested:
                    collected.append(
                        nested
                    )

                continue

        # ----------------------------------------------------
        # Object
        # ----------------------------------------------------

        text_value = getattr(
            item,
            "text",
            None,
        )

        if text_value:

            collected.append(
                str(text_value)
            )

            continue

        content = getattr(
            item,
            "content",
            None,
        )

        if content:

            nested = extract_from_items(
                content
            )

            if nested:
                collected.append(
                    nested
                )

    return "\n".join(
        collected
    ).strip()


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    "What is your return policy?",

    "How long does shipping take?",

    "Do you offer free shipping?",

    "Do you offer exchanges?",

    "How long does a refund take?",

    "Can I change my shipping address?",

    "What happens if I receive a damaged item?",

    "Can I cancel my order?",

    "What items cannot be returned?",

]


# ============================================================
# TEST RUNNER
# ============================================================

def main() -> None:

    print("\n")

    print("#" * 75)

    print(
        "NAARIRA PHASE A4 — POLICY RAG TEST"
    )

    print("#" * 75)

    for number, question in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        print("\n")

        print(
            f"TEST {number}/{len(TEST_QUERIES)}"
        )

        print(
            f"Question: {question}"
        )

        try:

            result = generate_policy_answer(
                question=question,
                top_k=3,
            )

            print("\nANSWER:")
            print(
                result["answer"]
            )

        except Exception as exc:

            print(
                "\nTEST FAILED"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: {exc}"
            )

    print("\n")

    print("#" * 75)

    print(
        "PHASE A4 TEST COMPLETE"
    )

    print("#" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()