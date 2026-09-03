"""Verify importing the cloud API does not load local ML runtimes."""

from __future__ import annotations

import os
import subprocess
import sys


def test_cloud_api_import_isolated_from_local_ml_dependencies():
    environment = os.environ.copy()
    environment.update(
        {
            "ENVIRONMENT": "production",
            "DEPLOYMENT_PROFILE": "cloud",
            "RETRIEVAL_PROVIDER": "qdrant",
            "GENERATION_PROVIDER": "groq",
            "RERANKER_PROVIDER": "disabled",
            "QDRANT_URL": "https://fixture.qdrant.test",
            "QDRANT_API_KEY": "fixture-secret",
            "QDRANT_DENSE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "BACKEND_API_TOKEN": "backend-secret",
            "CORS_ORIGINS": "https://fixture.streamlit.app",
        }
    )
    command = (
        "import sys; import nlp_academic_search.api.main; "
        "blocked={'torch','sentence_transformers','faiss','ollama','streamlit'}; "
        "assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
