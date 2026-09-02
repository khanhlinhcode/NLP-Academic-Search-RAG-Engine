"""
Semantic Search module using Sentence-BERT + FAISS.

Implements dense retrieval by encoding papers as embedding vectors
and finding nearest neighbors in the vector space.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.data.loader import Paper
from src.search.bm25_search import SearchResult


class SemanticSearcher:
    """
    Semantic search engine using Sentence-BERT embeddings + FAISS index.

    Encodes papers as dense vectors using a pretrained Sentence-BERT model,
    then uses FAISS for efficient approximate nearest neighbor search.

    Attributes:
        model: SentenceTransformer model instance.
        papers: List of Paper objects in the index.
        index: FAISS index for vector similarity search.
        embeddings: Pre-computed paper embeddings as numpy array.
    """

    def __init__(
        self,
        papers: List[Paper],
        model_name: Optional[str] = None,
        load_existing: bool = True,
    ):
        """
        Initialize semantic search with papers and embedding model.

        Args:
            papers: List of Paper objects to index.
            model_name: Name of the Sentence-BERT model. Defaults to config value.
            load_existing: Whether to try loading pre-computed embeddings from disk.
        """
        self.papers = papers
        model_name = model_name or settings.embedding.model_name

        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Try to load pre-computed embeddings
        embeddings_path = settings.data.embeddings_dir / "paper_embeddings.npy"
        index_path = settings.data.embeddings_dir / "faiss.index"

        if load_existing and embeddings_path.exists() and index_path.exists():
            print("📂 Loading pre-computed embeddings and FAISS index...")
            self.embeddings = np.load(str(embeddings_path))
            self.index = faiss.read_index(str(index_path))
            print(f"✅ Loaded {self.embeddings.shape[0]} embeddings (dim={self.embeddings.shape[1]})")
        else:
            print("🧮 Computing embeddings (this may take a while)...")
            self.embeddings = self._compute_embeddings()
            self.index = self._build_faiss_index(self.embeddings)
            print(f"✅ Semantic index built: {self.embeddings.shape[0]} vectors, dim={self.embeddings.shape[1]}")

    def _compute_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for all papers using Sentence-BERT.

        Returns:
            numpy array of shape (num_papers, embedding_dim).
        """
        texts = [paper.text for paper in self.papers]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build a FAISS index for inner product (cosine similarity) search.

        Uses IndexFlatIP since embeddings are L2-normalized,
        so inner product == cosine similarity.

        Args:
            embeddings: numpy array of shape (num_papers, embedding_dim).

        Returns:
            FAISS index ready for search.
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim on normalized vectors
        index.add(embeddings.astype(np.float32))
        return index

    def save_index(self, embeddings_path: Optional[Path] = None, index_path: Optional[Path] = None):
        """
        Save embeddings and FAISS index to disk.

        Args:
            embeddings_path: Path to save embeddings numpy file.
            index_path: Path to save FAISS index file.
        """
        settings.data.ensure_dirs()

        if embeddings_path is None:
            embeddings_path = settings.data.embeddings_dir / "paper_embeddings.npy"
        if index_path is None:
            index_path = settings.data.embeddings_dir / "faiss.index"

        np.save(str(embeddings_path), self.embeddings)
        faiss.write_index(self.index, str(index_path))
        print(f"💾 Saved embeddings to {embeddings_path}")
        print(f"💾 Saved FAISS index to {index_path}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search papers by semantic similarity.

        Encodes the query using the same model and finds nearest neighbors
        in the FAISS index.

        Args:
            query: User search query string.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult objects sorted by cosine similarity (descending).
        """
        # Encode query with the same model
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append(
                    SearchResult(paper=self.papers[idx], score=float(score))
                )

        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get the embedding vector for a query (used in hybrid search).

        Args:
            query: User search query string.

        Returns:
            Normalized embedding vector as numpy array.
        """
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get semantic similarity scores for all documents (used in hybrid search).

        Args:
            query: User search query string.

        Returns:
            numpy array of cosine similarity scores for all papers.
        """
        query_embedding = self.get_query_embedding(query)

        # Compute cosine similarity with all embeddings
        # Since both are L2-normalized, dot product = cosine similarity
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        return scores
