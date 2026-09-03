"""Lazy local Cross-Encoder reranking provider."""

from nlp_academic_search.search.reranker import Reranker


class LocalRerankerProvider(Reranker):
    provider_name = "local"
