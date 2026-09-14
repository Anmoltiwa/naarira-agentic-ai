import os
import json

from dotenv import load_dotenv
from google import genai

from rag.search import semantic_search


load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")


client = genai.Client(
    api_key=GEMINI_API_KEY
)

GENERATIVE_MODEL = "gemini-3.6-flash"


# --------------------------------------------------
# Build RAG Context
# --------------------------------------------------

def build_context(results):

    context_parts = []

    for index, product in enumerate(results, start=1):

        context_parts.append(
            f"""
PRODUCT {index}

Product ID:
{product["product_id"]}

Title:
{product["title"]}

Category:
{product["product_type"] or "Not specified"}

Similarity:
{product["similarity"]:.4f}

Product Information:
{product["content"]}

Product URL:
https://naarira.com/products/{product["handle"]}

--------------------------------------------------
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate RAG Answer
# --------------------------------------------------

def generate_rag_answer(query: str, top_k: int = 5):

    print("\n" + "=" * 60)
    print("NAARIRA RAG")
    print("=" * 60)

    print(f"\nCustomer Query: {query}")

    # --------------------------------------------------
    # Step 1: Semantic Search
    # --------------------------------------------------

    print("\nStep 1: Searching relevant products...")

    results = semantic_search(
        query=query,
        top_k=top_k
    )

    if not results:

        return {
            "answer": (
                "Sorry, I couldn't find any relevant "
                "products on Naarira."
            ),
            "products": [],
            "sources": []
        }

    # --------------------------------------------------
    # Step 2: Build Context
    # --------------------------------------------------

    print("\nStep 2: Building context...")

    context = build_context(results)

    # --------------------------------------------------
    # Step 3: Gemini Prompt
    # --------------------------------------------------

    print("\nStep 3: Preparing Gemini prompt...")

    prompt = f"""
You are Naarira's AI shopping assistant.

Naarira is a women's ethnic fashion store.

Your task is to answer the customer's question
using ONLY the retrieved product information below.

CUSTOMER QUESTION:
{query}


RETRIEVED PRODUCT INFORMATION:
{context}


STRICT GROUNDING RULES:

1. Use ONLY information explicitly present in the
   retrieved product information.

2. NEVER invent or assume:
   - product names
   - prices
   - colors
   - sizes
   - fabric
   - availability
   - discounts
   - features
   - specifications
   - delivery information
   - return information
   - stock information

3. Do not combine information from different products.

4. Do not infer a product property from its name.

5. If a requested property is not present in the
   retrieved information, say that the information
   is not available.

6. Recommend ONLY products that appear in the
   retrieved product information.

7. Do not create product URLs yourself.
   Product URLs are already provided in the context.

8. Do not use Markdown links.

9. Do not include URLs in the answer.

10. Do not say that the customer can place an order
    unless the retrieved information explicitly
    provides such information.

11. Do not mention internal technical concepts such as:
    - RAG
    - embeddings
    - vectors
    - pgvector
    - semantic search
    - prompts
    - database
    - Gemini

12. Keep the response concise and customer-friendly.

13. If multiple products match, mention the most
    relevant products from the retrieved information.

14. If the retrieved products do not provide enough
    information to answer the question, clearly say so.

Now answer the customer naturally.
"""

    # --------------------------------------------------
    # Step 4: Gemini
    # --------------------------------------------------

    print("\nStep 4: Generating Gemini answer...")

    interaction = client.interactions.create(
        model=GENERATIVE_MODEL,
        input=prompt
    )

    answer = interaction.output_text.strip()

    # --------------------------------------------------
    # Step 5: Structured Product Sources
    # --------------------------------------------------

    products = []

    sources = []

    for product in results:

        product_data = {
            "product_id": product["product_id"],
            "title": product["title"],
            "handle": product["handle"],
            "category": product["product_type"],
            "similarity": round(
                product["similarity"],
                4
            ),
            "url": (
                f"https://naarira.com/products/"
                f"{product['handle']}"
            )
        }

        products.append(product_data)

        sources.append({
            "product_id": product["product_id"],
            "title": product["title"],
            "similarity": round(
                product["similarity"],
                4
            ),
            "url": (
                f"https://naarira.com/products/"
                f"{product['handle']}"
            )
        })

    return {
        "answer": answer,
        "products": products,
        "sources": sources
    }


# --------------------------------------------------
# Local Testing
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("NAARIRA RAG CHAT")
    print("=" * 60)

    query = input(
        "\nAsk something about Naarira products: "
    ).strip()

    if not query:

        print("\nPlease enter a question.")

    else:

        result = generate_rag_answer(
            query=query,
            top_k=5
        )

        print("\n" + "-" * 60)
        print("FINAL ANSWER")
        print("-" * 60)

        print("\n" + result["answer"])

        print("\n" + "-" * 60)
        print("PRODUCTS")
        print("-" * 60)

        for product in result["products"]:

            print(
                f"\n{product['title']}"
            )

            print(
                f"Similarity: "
                f"{product['similarity']}"
            )

            print(
                f"URL: "
                f"{product['url']}"
            )

        print("\n" + "=" * 60)
        print("RAG COMPLETE")
        print("=" * 60)