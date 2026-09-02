"""
RAG API routes.

Provides endpoints for Retrieval-Augmented Generation (question answering).
"""

import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import AskRequest, AskResponse, SourceReference
from src.rag.prompt_builder import build_rag_prompt, build_source_list

router = APIRouter(prefix="/ask", tags=["RAG"])


@router.post("", response_model=AskResponse, summary="Ask a Question (RAG)")
async def ask_question(request: AskRequest):
    """
    Answer a question using Retrieval-Augmented Generation (RAG).

    Pipeline:
    1. Retrieve relevant papers using hybrid search
    2. Optionally re-rank with Cross-Encoder
    3. Build a prompt with paper contexts
    4. Generate answer with LLM (Ollama)
    5. Return answer with source references
    """
    from src.api.main import app_state

    start_time = time.time()

    # Step 1: Retrieve papers using hybrid search
    candidate_k = request.top_k * 4 if request.use_reranker else request.top_k
    results = app_state.hybrid_searcher.search(
        request.question, top_k=candidate_k
    )

    # Step 2: Optionally rerank
    if request.use_reranker and app_state.reranker is not None:
        results = app_state.reranker.rerank(
            request.question, results, top_k=request.top_k
        )
    else:
        results = results[: request.top_k]

    # Step 3: Build RAG prompt
    papers = [r.paper for r in results]
    prompt = build_rag_prompt(request.question, papers)

    # Step 4: Generate answer
    answer = app_state.rag_generator.generate(prompt)

    # Step 5: Build response
    source_list = build_source_list(papers)
    sources = [SourceReference(**s) for s in source_list]

    latency_ms = (time.time() - start_time) * 1000

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_method="hybrid" + (" + reranker" if request.use_reranker else ""),
        latency_ms=round(latency_ms, 2),
    )


@router.post("/stream", summary="Ask a Question (Streaming RAG)")
async def ask_question_stream(request: AskRequest):
    """
    Stream a RAG answer token by token.

    Same pipeline as /ask but returns a streaming response
    for real-time display in the frontend.
    """
    from src.api.main import app_state

    # Retrieve and rerank
    candidate_k = request.top_k * 4 if request.use_reranker else request.top_k
    results = app_state.hybrid_searcher.search(
        request.question, top_k=candidate_k
    )

    if request.use_reranker and app_state.reranker is not None:
        results = app_state.reranker.rerank(
            request.question, results, top_k=request.top_k
        )
    else:
        results = results[: request.top_k]

    # Build prompt
    papers = [r.paper for r in results]
    prompt = build_rag_prompt(request.question, papers)

    # Stream response
    def generate():
        for token in app_state.rag_generator.generate_stream(prompt):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")
