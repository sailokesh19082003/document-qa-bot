"""
main.py
-------
Command-line interface for the Document Q&A Bot.

Run with:
    python src/main.py

This satisfies the assignment's minimum requirement: "the bot must run
from the command line as an interactive loop." It shows the answer AND
the source chunks used for every question.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import GEMINI_API_KEY, DB_DIR, COLLECTION_NAME
from query import query_rag_pipeline


def print_banner():
    print("=" * 60)
    print(" Health & Nutrition Document Q&A Bot (RAG-powered)")
    print(" Type your question below, or 'exit' to quit.")
    print("=" * 60)


def check_setup():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Create a .env file (see README.md).")
        return False
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("ERROR: No vector database found. Run 'python src/ingest.py' first.")
        return False
    return True


def main():
    if not check_setup():
        return

    print_banner()

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        print("\nSearching documents and generating answer...\n")
        result = query_rag_pipeline(query)

        print("-" * 60)
        print("ANSWER:")
        print(result["answer"])
        print("-" * 60)

        if result["citations"]:
            print("\nSOURCES USED:")
            seen = set()
            for cite, snippet in zip(result["citations"], result["raw_context"]):
                if cite not in seen:
                    seen.add(cite)
                    preview = snippet[:150] + ("..." if len(snippet) > 150 else "")
                    print(f"  [{cite}] -> \"{preview}\"")


if __name__ == "__main__":
    main()
