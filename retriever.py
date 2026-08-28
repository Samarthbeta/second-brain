"""
Second Brain — Retrieval & Chat Generation (RAG)
=================================================
When the user asks a question:
  1. Extract 2-3 search keywords from the question via LiteLLM
  2. Query SQLite FTS5 for matching transcripts
  3. Synthesize a final answer using LiteLLM with the retrieved context
"""

from __future__ import annotations

import asyncio
import os

import litellm

litellm.suppress_debug_info = True

from database import get_connection, search_fts


# ─── Keyword Extraction ───

async def extract_keywords(question: str, num_keywords: int = 3) -> list[str]:
    """
    Use a fast LLM to extract 2-3 search keywords from the user's question.
    These keywords are used to query the FTS5 index.
    """
    # Determine available model
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    model = "gemini/gemini-2.0-flash" if gemini_key else "groq/llama-3.1-8b-instant"

    if not groq_key and not gemini_key:
        # No LLM available — fall back to simple word extraction
        return _simple_keyword_extraction(question)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Extract exactly {num_keywords} search keywords from this question. "
                        "Focus on nouns and key concepts. Return ONLY the keywords separated by commas. "
                        "No explanation, no punctuation other than commas."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=30,
        )
        raw = response.choices[0].message.content.strip()
        keywords = [kw.strip().strip('"').strip("'") for kw in raw.split(",")]
        keywords = [kw for kw in keywords if len(kw) > 1][:num_keywords]
        return keywords if keywords else _simple_keyword_extraction(question)

    except Exception:
        return _simple_keyword_extraction(question)


def _simple_keyword_extraction(question: str) -> list[str]:
    """Fallback: extract meaningful words from the question."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "what", "which", "who",
        "whom", "this", "that", "these", "those", "i", "me", "my", "we",
        "about", "it", "its", "you", "your", "he", "she", "they", "them",
    }
    words = question.lower().split()
    keywords = [w for w in words if len(w) > 2 and w.isalnum() and w not in stopwords]
    return keywords[:3] if keywords else ["general"]


# ─── Context Retrieval ───

def retrieve_context(question: str, keywords: list[str], max_chars: int = 8000) -> str:
    """
    Search the FTS5 index using the extracted keywords.
    Returns a concatenated context block of the most relevant transcripts.
    """
    conn = get_connection()
    try:
        # Try FTS search with the full keyword string
        results = search_fts(conn, " ".join(keywords), limit=3)

        if not results:
            # Fallback: try individual keywords
            for kw in keywords:
                results = search_fts(conn, kw, limit=3)
                if results:
                    break

        if not results:
            return ""

        # Concatenate transcripts, respecting max_chars limit
        context_parts = []
        total = 0
        for r in results:
            title = r["title"]
            transcript = r["transcript"]
            duration = r["duration"]
            entry = f"[{title}] ({duration})\n{transcript}\n"
            if total + len(entry) > max_chars:
                remaining = max_chars - total
                entry = entry[:remaining] + "...\n"
                context_parts.append(entry)
                break
            context_parts.append(entry)
            total += len(entry)

        return "\n---\n".join(context_parts)
    finally:
        conn.close()


# ─── Chat Generation (RAG Synthesis) ───

async def chat_with_brain(question: str, history: list[dict] | None = None) -> dict:
    """
    Full RAG pipeline:
      1. Extract keywords from the question
      2. Retrieve matching transcripts from SQLite FTS5
      3. Synthesize an answer using LiteLLM

    Returns {"answer": str, "sources": list[dict], "keywords": list[str]}.
    """
    # Step 1: Extract keywords
    keywords = await extract_keywords(question)

    # Step 2: Retrieve context
    context = retrieve_context(question, keywords)

    # Step 3: Synthesize answer
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model = "gemini/gemini-2.0-flash" if gemini_key else "groq/llama-3.1-8b-instant"

    sources = []
    if not context:
        answer = (
            "I couldn't find any matching transcripts in your Second Brain. "
            "Try uploading some audio or video files first, or rephrase your question."
        )
    else:
        # Build the system prompt with context
        system_prompt = (
            "You are the user's Second Brain — an AI assistant that answers questions "
            "based on their personal media library (transcripts of videos and audio files). "
            "Use the provided transcript context to answer the user's question accurately. "
            "If the context doesn't contain enough information, say so honestly. "
            "Cite which transcript(s) you used. Be concise but thorough."
        )

        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add history if present
        if history:
            for msg in history[-6:]:  # Last 6 messages for context
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the retrieval context + user question
        user_prompt = (
            f"## Retrieved Transcript Context\n\n{context}\n\n"
            f"## User Question\n{question}"
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            if not groq_key and not gemini_key:
                answer = (
                    "No API keys configured. Please add GROQ_API_KEY or GEMINI_API_KEY "
                    "to your .env file to enable chat."
                )
            else:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error generating response: {e}"

        # Parse sources from context
        conn = get_connection()
        try:
            for kw in keywords:
                results = search_fts(conn, kw, limit=3)
                for r in results:
                    if r["id"] not in [s["id"] for s in sources]:
                        sources.append({
                            "id": r["id"],
                            "title": r["title"],
                            "duration": r["duration"],
                            "relevance": r.get("relevance", 0),
                        })
        finally:
            conn.close()

    return {
        "answer": answer,
        "sources": sources[:5],
        "keywords": keywords,
    }
