"""
Text preprocessing module.

Provides text cleaning and tokenization utilities for BM25 and embedding pipelines.
"""

import re
import string
from typing import List


def clean_text(text: str) -> str:
    """
    Clean raw text for processing.

    - Normalizes whitespace
    - Removes excessive special characters
    - Preserves meaningful punctuation

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    # Normalize unicode
    text = text.strip()

    # Remove LaTeX commands (common in arXiv papers)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Remove math mode markers
    text = re.sub(r"\$[^$]*\$", "", text)

    # Remove citations like [1], [2,3]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.

    Simple whitespace + punctuation tokenization with lowercasing.
    Removes common stop words for better BM25 performance.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    # Lowercase and clean
    text = clean_text(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split on whitespace
    tokens = text.split()

    # Remove stop words (minimal set for scientific text)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or", "if",
        "while", "that", "this", "these", "those", "it", "its", "we", "our",
        "they", "their", "them", "what", "which", "who", "whom", "whose",
    }

    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    return tokens


def prepare_text_for_embedding(title: str, abstract: str) -> str:
    """
    Combine and clean title + abstract for embedding generation.

    Args:
        title: Paper title.
        abstract: Paper abstract.

    Returns:
        Combined cleaned text ready for the embedding model.
    """
    title = clean_text(title)
    abstract = clean_text(abstract)
    return f"{title}. {abstract}"
