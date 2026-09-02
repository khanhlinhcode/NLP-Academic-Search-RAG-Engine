"""
RAG Prompt Builder module.

Constructs prompts for the LLM by combining the user's question
with relevant paper contexts retrieved by the search pipeline.
"""

from typing import List

from src.data.loader import Paper


# System prompt that instructs the LLM how to behave
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer science and AI.
Your task is to answer questions based ONLY on the provided academic papers.

Rules:
1. Base your answer ONLY on the information found in the provided papers.
2. Cite your sources using [1], [2], etc. corresponding to the paper numbers.
3. If the papers don't contain enough information to answer the question, say so explicitly.
4. Be precise, technical, and informative.
5. Structure your answer clearly with key points."""


def build_rag_prompt(question: str, papers: List[Paper]) -> str:
    """
    Build a RAG prompt by combining the question with paper contexts.

    Creates a structured prompt that:
    - Lists retrieved papers with their titles and abstracts
    - Instructs the LLM to answer using only the provided context
    - Requires citation of sources

    Args:
        question: The user's question.
        papers: List of relevant Paper objects retrieved by the search pipeline.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    # Build context from papers
    context_parts = []
    for i, paper in enumerate(papers, start=1):
        context_parts.append(
            f"[{i}] Title: {paper.title}\n"
            f"    Abstract: {paper.abstract}\n"
        )

    context = "\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

--- RETRIEVED PAPERS ---
{context}
--- END OF PAPERS ---

Question: {question}

Answer (cite sources using [1], [2], etc.):"""

    return prompt


def build_source_list(papers: List[Paper]) -> List[dict]:
    """
    Build a structured list of source references.

    Args:
        papers: List of Paper objects used as context.

    Returns:
        List of source dictionaries with id, title, and index.
    """
    sources = []
    for i, paper in enumerate(papers, start=1):
        sources.append({
            "index": i,
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "category": paper.category,
            "year": paper.year,
        })
    return sources
