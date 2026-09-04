"""Interactive Streamlit workspace for academic search and streaming RAG."""

from __future__ import annotations

import os
from collections.abc import Iterable

import streamlit as st

from nlp_academic_search.ui.api_client import (
    DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
    AcademicSearchClient,
    APIError,
    parse_request_timeout,
)
from nlp_academic_search.ui.security import escape_html, safe_external_url
from nlp_academic_search.ui.styles import stylesheet


def configure_page() -> None:
    """Apply page metadata and the visual theme on every Streamlit rerun."""
    st.set_page_config(
        page_title="Academic Search & RAG",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(stylesheet(), unsafe_allow_html=True)


def configured_api_base_url() -> str:
    configured = os.getenv("API_BASE_URL")
    if not configured:
        try:
            configured = str(st.secrets["API_BASE_URL"])
        except (KeyError, FileNotFoundError):
            configured = None
    return (configured or "http://localhost:8000").rstrip("/")


def configured_backend_token() -> str | None:
    configured = os.getenv("BACKEND_API_TOKEN")
    if not configured:
        try:
            configured = str(st.secrets["BACKEND_API_TOKEN"])
        except (KeyError, FileNotFoundError):
            configured = None
    return configured or None


def configured_api_request_timeout() -> float:
    configured: object = os.getenv("API_REQUEST_TIMEOUT_SECONDS")
    if configured is None:
        try:
            configured = st.secrets["API_REQUEST_TIMEOUT_SECONDS"]
        except (KeyError, FileNotFoundError):
            configured = DEFAULT_API_REQUEST_TIMEOUT_SECONDS
    return parse_request_timeout(configured)


@st.cache_resource
def get_client(
    base_url: str, api_token: str | None, timeout_seconds: float
) -> AcademicSearchClient:
    return AcademicSearchClient(base_url, api_token=api_token, timeout=timeout_seconds)


@st.cache_data(ttl=8, show_spinner=False)
def get_health(base_url: str, api_token: str | None, timeout_seconds: float) -> dict:
    return get_client(base_url, api_token, timeout_seconds).health()


def safe(value: object) -> str:
    return escape_html(value)


def authors_text(authors: Iterable[str], limit: int = 4) -> str:
    names = list(authors)
    if not names:
        return "Unknown authors"
    visible = names[:limit]
    suffix = f" +{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(visible) + suffix


def source_url(record: dict) -> str | None:
    return safe_external_url(record)


def categories_text(record: dict) -> str:
    categories = record.get("categories") or []
    return ", ".join(categories) if categories else "Category unavailable"


def format_duration(milliseconds: object) -> str:
    try:
        value = float(str(milliseconds))
    except (TypeError, ValueError):
        return "—"
    if value < 1000:
        return f"{value:.0f} ms"
    return f"{value / 1000:.2f} s"


def render_masthead(health: dict | None, health_error: str | None) -> None:
    connected = health is not None
    ready = bool(health and health.get("status") == "ready")
    status_label = "READY" if ready else "DEGRADED" if connected else "OFFLINE"
    status_class = "ready" if ready else "offline"
    papers = f"{health.get('total_papers', 0):,}" if health else "—"
    model = safe((health or {}).get("models", {}).get("llm", "—"))
    generation = "AVAILABLE" if health and health.get("generation_available") else "UNAVAILABLE"
    generation_provider = safe(
        str((health or {}).get("providers", {}).get("generation", "generation")).upper()
    )
    st.markdown(
        f"""
        <header class="masthead">
          <div>
            <h1>NLP Academic Search &amp; RAG Engine</h1>
            <div class="strap">Retrieve &nbsp;·&nbsp; Read &nbsp;·&nbsp; Reason</div>
          </div>
          <dl class="engine-meta">
            <dt>ENGINE STATUS</dt><dd class="{status_class}">[ {status_label} ]</dd>
            <dt>CORPUS</dt><dd>{papers} papers</dd>
            <dt>{generation_provider}</dt><dd>{generation}</dd>
            <dt>MODEL</dt><dd>{model}</dd>
          </dl>
        </header>
        """,
        unsafe_allow_html=True,
    )
    if health_error:
        st.warning(health_error, icon=None)
        if st.button("Retry backend connection", key="retry-backend", type="primary"):
            get_health.clear()
            st.rerun()


def render_pipeline(health: dict | None, *, label: str) -> None:
    generation_ready = bool(
        health and health.get("generation_available", health.get("ollama_available"))
    )
    providers = (health or {}).get("providers", {})
    retrieval_name = str(providers.get("retrieval", "local")).upper()
    generation_name = str(providers.get("generation", "ollama")).upper()
    verification_enabled = bool(
        health and health.get("providers", {}).get("verification") != "disabled"
    )
    verification_ready = bool(health and health.get("verification_available"))
    with st.expander(label, expanded=False):
        st.markdown(
            '<p class="pipeline-note">Five inspectable stages connect the query to its evidence.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <ol class="pipeline-steps">
              <li><span>01</span><div><strong>Understand query</strong><small>Validate and normalize intent</small></div></li>
              <li><span>02</span><div><strong>Retrieve candidates</strong><small>{safe(retrieval_name)} dense + sparse</small></div></li>
              <li><span>03</span><div><strong>Fuse rankings</strong><small>Reciprocal rank fusion</small></div></li>
              <li><span>04</span><div><strong>Refine evidence</strong><small>Optional Cross-Encoder</small></div></li>
              <li><span>05</span><div><strong>Ground answer</strong><small>{safe(generation_name)} + evidence verification</small></div></li>
            </ol>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="rail-title">System readiness</h3>', unsafe_allow_html=True)
        checks = [
            (health is not None, "API connected"),
            (bool(health and health.get("total_papers")), "Paper index loaded"),
            (generation_ready, f"{generation_name} model available"),
        ]
        if verification_enabled:
            checks.append((verification_ready, "Semantic verifier model reachable"))
        for ok, check_label in checks:
            state = "Ready" if ok else "Check"
            color = "ready" if ok else "offline"
            st.markdown(
                f'<div class="readiness-row"><span class="status-dot {color}"></span>'
                f"<span>{safe(check_label)}</span><strong>{state}</strong></div>",
                unsafe_allow_html=True,
            )


def render_benchmark() -> None:
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="rail-title">Evaluation status</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="source-meta">A labelled repository fixture is available with
        <code>make eval-retrieval</code>. Standard SciFact/BEIR quality remains
        <strong>not evaluated</strong>.</p>
        """,
        unsafe_allow_html=True,
    )


def render_paper(paper: dict, rank: int, *, expanded: bool = False) -> None:
    title = safe(paper.get("title"))
    authors = safe(authors_text(paper.get("authors", [])))
    categories = safe(categories_text(paper))
    year = safe(paper.get("year"))
    score = float(paper.get("score", 0.0))
    abstract = safe(paper.get("abstract"))
    score_type = safe(paper.get("score_type", "score")).replace("_", " ")
    provenance = source_url(paper)
    identifier = (
        f"arXiv:{safe(paper.get('arxiv_id'))}"
        if paper.get("arxiv_id")
        else f"DOI:{safe(paper.get('doi'))}"
        if paper.get("doi")
        else safe(paper.get("source"))
    )
    st.markdown(
        f"""
        <article class="paper-entry">
          <div class="paper-head">
            <div class="paper-rank">{rank}</div>
            <div class="paper-title">{title}</div>
            <div class="paper-score">{score_type} {score:.4f}</div>
          </div>
          <div class="paper-meta-chips">
            <span class="meta-chip highlight">{categories}</span>
            <span class="meta-chip">{year}</span>
            <span class="meta-chip">{identifier}</span>
            <span class="meta-chip">{authors}</span>
          </div>
          {f'<div class="paper-abstract">{abstract}</div>' if expanded else ""}
        </article>
        """,
        unsafe_allow_html=True,
    )
    if not expanded:
        with st.expander("Read abstract"):
            st.write(paper.get("abstract", "No abstract available."))
    action_a, action_b, _ = st.columns([1.15, 1, 4])
    with action_a:
        if provenance:
            st.link_button("Open source", provenance, use_container_width=True)
    with action_b:
        if st.button("Inspect", key=f"inspect-{paper.get('id')}-{rank}", use_container_width=True):
            st.session_state.selected_paper = paper


def render_selected_paper(paper: dict | None) -> None:
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="rail-title">Selected paper</h3>', unsafe_allow_html=True)
    if not paper:
        st.markdown(
            '<p class="source-meta">Choose “Inspect” on a result to pin its metadata here.</p>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="source-title">{safe(paper.get("title"))}</div>
        <div class="source-meta">
          {safe(authors_text(paper.get("authors", []), limit=6))}<br>
          {safe(categories_text(paper))} · {safe(paper.get("year"))}<br>
          {safe(paper.get("arxiv_id") or paper.get("doi") or paper.get("source"))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    provenance = source_url(paper)
    if provenance:
        st.link_button("Read original", provenance, use_container_width=True)


def render_search(client: AcademicSearchClient, health: dict | None) -> None:
    def execute_search(parameters: dict) -> None:
        st.session_state.search_response = client.search(**parameters)
        results = st.session_state.search_response.get("results", [])
        st.session_state.selected_paper = results[0] if results else None

    with st.form("search-form", border=False):
        query_col, action_col = st.columns([5, 1.15], vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Query",
                value=st.session_state.get("search_query", ""),
                placeholder="e.g. dense retrieval for open-domain question answering",
                max_chars=500,
            )
        with action_col:
            submitted = st.form_submit_button(
                "Search papers", type="primary", use_container_width=True
            )
        setting_a, setting_b, setting_c = st.columns([2.1, 1.5, 4.4], vertical_alignment="bottom")
        with setting_a:
            method = st.selectbox(
                "Method",
                options=["hybrid", "bm25", "semantic"],
                format_func=lambda value: {
                    "hybrid": "Hybrid · RRF",
                    "bm25": "BM25 · keyword",
                    "semantic": "SBERT · semantic",
                }[value],
            )
        with setting_b:
            top_k = st.selectbox("Results per page", options=[5, 10, 15, 20, 30, 50], index=1)
        with setting_c:
            st.markdown(
                '<p class="method-explainer"><strong>Hybrid recommended.</strong> '
                "Combines exact terminology with semantic similarity.</p>",
                unsafe_allow_html=True,
            )
        with st.expander("Filter metadata"):
            filter_a, filter_b, filter_c, filter_d, filter_e = st.columns(5)
            category = filter_a.text_input("Category", placeholder="cs.CL")
            year_from = filter_b.number_input(
                "From year", min_value=1900, max_value=2100, value=None, step=1
            )
            year_to = filter_c.number_input(
                "To year", min_value=1900, max_value=2100, value=None, step=1
            )
            author = filter_d.text_input("Author contains", placeholder="Vaswani")
            source = filter_e.text_input("Source", placeholder="arxiv")

    if submitted:
        query_text = (query or "").strip()
        if not query_text:
            st.error("Enter a paper topic or research question before searching.", icon=None)
        else:
            st.session_state.search_query = query_text
            try:
                with st.spinner("Retrieving and fusing ranked results…"):
                    parameters = {
                        "query": query_text,
                        "method": method,
                        "top_k": top_k,
                        "category": category,
                        "year_from": year_from,
                        "year_to": year_to,
                        "author": author,
                        "source": source,
                        "offset": 0,
                    }
                    execute_search(parameters)
                    st.session_state.search_request = parameters
            except APIError as exc:
                st.error(str(exc), icon=None)

    response = st.session_state.get("search_response")
    main_col, rail_col = st.columns([1.75, 1], gap="large")
    with main_col:
        if response:
            for warning in response.get("warnings", []):
                st.warning(str(warning), icon=None)
            st.markdown(
                f'<div class="result-summary"><strong>{response.get("total_results", 0)} papers</strong>'
                f"<span>{safe(str(response.get('method', 'hybrid')).upper())}</span>"
                f"<span>Retrieved in {format_duration(response.get('latency_ms'))}</span></div>",
                unsafe_allow_html=True,
            )
            results = response.get("results", [])
            if not results:
                st.info(
                    "No papers matched this query and metadata filter. Broaden the filters or try BM25.",
                    icon=None,
                )
            for rank, paper in enumerate(results, start=1):
                absolute_rank = response.get("offset", 0) + rank
                render_paper(paper, absolute_rank, expanded=(rank == 1))
            if results:
                previous_col, page_col, next_col = st.columns([1, 2, 1])
                offset = int(response.get("offset", 0))
                page_size = int(response.get("page_size", len(results)))
                with previous_col:
                    if st.button("Previous", disabled=offset == 0, use_container_width=True):
                        parameters = dict(st.session_state.search_request)
                        parameters["offset"] = max(0, offset - page_size)
                        execute_search(parameters)
                        st.session_state.search_request = parameters
                        st.rerun()
                with page_col:
                    st.caption(f"Page {offset // max(1, page_size) + 1}")
                with next_col:
                    if st.button(
                        "Next",
                        disabled=not response.get("has_more", False),
                        use_container_width=True,
                    ):
                        parameters = dict(st.session_state.search_request)
                        parameters["offset"] = offset + page_size
                        execute_search(parameters)
                        st.session_state.search_request = parameters
                        st.rerun()
        else:
            st.markdown(
                """
                <section class="empty-sheet">
                  <h3>Begin with a research concept</h3>
                  <p>Search an exact method such as “reciprocal rank fusion,” or
                  describe the meaning you need: “methods that reduce hallucination
                  with retrieved evidence.”</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
    with rail_col:
        with st.container(key="search_rail"):
            render_selected_paper(st.session_state.get("selected_paper"))
            render_pipeline(health, label="How this search works")
            render_benchmark()


def render_sources(sources: list, metadata: dict | None = None) -> None:
    source_count = len(sources)
    st.markdown(
        '<div class="ledger-heading"><h3 class="rail-title">Evidence ledger</h3>'
        f"<span>{source_count} {'source' if source_count == 1 else 'sources'}</span></div>",
        unsafe_allow_html=True,
    )
    cited_indices = set((metadata or {}).get("citation_validation", {}).get("cited_indices", []))
    if metadata:
        latencies = metadata.get("latencies") or {}
        retrieval = metadata.get("retrieval_ms", latencies.get("retrieval_ms"))
        generation = latencies.get("generation_ms")
        verification = latencies.get("verification_ms", metadata.get("verification_latency_ms"))
        total = latencies.get("total_ms", metadata.get("latency_ms"))
        timing = []
        if retrieval is not None:
            timing.append(f"retrieve {format_duration(retrieval)}")
        if generation is not None:
            timing.append(f"generate {format_duration(generation)}")
        if verification:
            timing.append(f"verify {format_duration(verification)}")
        if total is not None:
            timing.append(f"total {format_duration(total)}")
        method = safe(str(metadata.get("retrieval_method", "hybrid")).upper())
        st.markdown(
            f'<div class="ledger-timing"><strong>{method}</strong>'
            f"<span>{' · '.join(timing) if timing else 'Timing pending'}</span></div>",
            unsafe_allow_html=True,
        )
    if not sources:
        terminal = bool((metadata or {}).get("answer_status"))
        empty_title = "No evidence retrieved" if terminal else "Waiting for evidence"
        empty_detail = (
            "No indexed paper met the evidence threshold for this question."
            if terminal
            else "Retrieved papers appear here before generation begins."
        )
        st.markdown(
            f'<div class="evidence-empty"><strong>{empty_title}</strong>'
            f"<span>{empty_detail}</span></div>",
            unsafe_allow_html=True,
        )
        return
    for source in sources:
        source_index = source.get("index")
        cited = source_index in cited_indices
        identifier = (
            f"arXiv:{safe(source.get('arxiv_id'))}"
            if source.get("arxiv_id")
            else f"DOI:{safe(source.get('doi'))}"
            if source.get("doi")
            else safe(source.get("source"))
        )
        link = source_url(source)
        link_html = (
            f'<a href="{safe(link)}" target="_blank" rel="noopener">Open source</a>'
            if link
            else "No external source link"
        )
        cited_badge = (
            '<span class="cited-label"><span class="status-dot ready"></span> Cited</span>'
            if cited
            else ""
        )
        st.markdown(
            f"""
            <div class="source-item {"is-cited" if cited else ""}">
              <div class="source-heading">
                <span class="source-index">[{safe(source_index)}]</span>
                <span class="source-title">{safe(source.get("title"))}</span>
                {cited_badge}
              </div>
              <div class="source-meta">
                {safe(authors_text(source.get("authors", []), limit=3))}<br>
                {safe(categories_text(source))} · {safe(source.get("year"))} · {identifier}<br>
                {link_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_saved_answer(entry: dict) -> None:
    st.markdown('<div class="content-label">Research question</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="question">{safe(entry["question"])}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="answer-label">Answer grounded in retrieved papers</div>',
        unsafe_allow_html=True,
    )
    render_citation_verdict(entry.get("metadata", {}))
    st.markdown(entry.get("answer", ""))


def verification_failure_detail(metadata: dict) -> str:
    """Return actionable copy without exposing provider response details."""
    reason = metadata.get("failure_reason")
    if not isinstance(reason, str):
        reason = None
    details = {
        "SemanticVerificationInvalidRequest": (
            "The verifier request was rejected; check the backend verifier contract"
        ),
        "SemanticVerificationInvalidResponse": (
            "The verifier returned an invalid structured response; check provider compatibility"
        ),
        "SemanticVerificationAuthenticationError": (
            "Verifier authentication failed; check the backend provider credentials"
        ),
        "SemanticVerificationRateLimited": (
            "The verifier is rate-limited; check provider quota and retry later"
        ),
        "SemanticVerificationTimeout": "The verifier timed out; retry the request later",
        "SemanticVerificationUnavailable": (
            "The verifier is temporarily unavailable; retry the request later"
        ),
        "semantic_assessment_failed": (
            "One or more answer claims could not be supported by retrieved evidence"
        ),
    }
    if reason is not None and reason in details:
        return details[reason]
    return "The generated draft could not be verified against the retrieved evidence"


def render_citation_verdict(metadata: dict) -> None:
    citation = metadata.get("citation_validation") or {}
    if not citation:
        return
    status = metadata.get("answer_status")
    semantic = metadata.get("semantic_validation") or {}
    if status == "verified" and semantic.get("valid"):
        label = "Answer ready"
        detail = "Evidence verified — every factual claim has source-backed evidence"
        css_class = "is-complete"
        dot_class = "ready"
    elif status in {"refused_unverified", "refused_insufficient_context"}:
        label = "Answer withheld"
        detail = verification_failure_detail(metadata)
        css_class = "is-warning"
        dot_class = "warning"
    elif status == "verification_unavailable":
        label = "Semantic verification unavailable"
        detail = "Citation structure passed, but evidence support was not verified"
        css_class = "is-warning"
        dot_class = "warning"
    elif citation.get("valid"):
        label = "Citation format valid"
        detail = "Semantic verification is unavailable or disabled"
        css_class = "is-warning"
        dot_class = "warning"
    else:
        label = "Needs citation review"
        detail = "One or more factual sentences lack a valid source"
        css_class = "is-warning"
        dot_class = "warning"
    st.markdown(
        f'<div class="pipeline-state {css_class}"><span class="status-dot {dot_class}"></span>'
        f"<strong>{label}</strong><span>{detail}</span></div>",
        unsafe_allow_html=True,
    )


def render_ask(client: AcademicSearchClient, health: dict | None) -> None:
    history = st.session_state.setdefault("ask_history", [])
    rag_ready = bool(
        health
        and health.get("rag_enabled")
        and health.get("generation_available", health.get("ollama_available"))
        and (not health.get("verification_required") or health.get("verification_available"))
    )
    reranker_available = (health or {}).get("providers", {}).get("reranker") != "disabled"
    with st.form("ask-form", border=False):
        question = st.text_area(
            "Ask the literature",
            value="",
            placeholder="What evidence do the indexed papers provide about…?",
            height=92,
            max_chars=1000,
        )
        controls, action = st.columns([4, 1], vertical_alignment="bottom")
        with controls:
            top_k_col, reranker_col = st.columns([1, 2], vertical_alignment="bottom")
            with top_k_col:
                top_k = st.selectbox("Evidence papers", options=[3, 5, 8, 10, 15, 20], index=1)
            with reranker_col:
                use_reranker = st.toggle(
                    "Cross-Encoder reranker",
                    value=False,
                    disabled=not reranker_available,
                    help="Disabled by default until a leakage-free benchmark demonstrates a gain.",
                )
        with action:
            submitted = st.form_submit_button(
                "Ask", type="primary", use_container_width=True, disabled=not rag_ready
            )

    if not rag_ready:
        verification_blocked = bool(
            health
            and health.get("verification_required")
            and not health.get("verification_available")
        )
        unavailable_detail = (
            "Question answering is withheld because semantic verification is required but its "
            "provider is unavailable. Search remains available; verify the verifier model and "
            "provider quota."
            if verification_blocked
            else "Question answering is unavailable because the configured generation provider "
            "is not ready. Search remains available; verify the provider key, model and quota."
        )
        st.info(
            unavailable_detail,
            icon=None,
        )

    answer_col, evidence_col = st.columns([1.65, 1], gap="large")
    if submitted and len(question.strip()) < 5:
        st.error(
            "Write a question of at least five characters before asking the literature.",
            icon=None,
        )
    elif submitted:
        sources: list = []
        metadata: dict = {}
        error_message: str | None = None
        warnings: list[str] = []
        draft_answer = ""
        replacement_answer: str | None = None
        final_answer: str | None = None
        done_received = False
        history_committed = False
        with evidence_col:
            with st.container(key="evidence_rail"):
                evidence_loading_slot = st.empty()
                evidence_content_slot = st.empty()
                evidence_loading_slot.markdown(
                    '<div class="evidence-empty"><strong>Retrieving evidence…</strong>'
                    "<span>Relevant papers will appear before generation begins.</span></div>",
                    unsafe_allow_html=True,
                )
        with answer_col, st.container(key="answer_sheet"):
            st.markdown(
                '<div class="content-label">Research question</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="question">{safe(question.strip())}</div>',
                unsafe_allow_html=True,
            )
            stage_slot = st.empty()
            stage_slot.markdown(
                '<div class="pipeline-state"><span class="status-dot ready"></span>'
                "<strong>Retrieving evidence</strong><span>Running</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="answer-label">Draft answer · validation pending</div>',
                unsafe_allow_html=True,
            )
            answer_slot = st.empty()

            try:
                for event in client.stream_answer(
                    question.strip(), top_k=top_k, use_reranker=use_reranker
                ):
                    event_type = event.get("event")
                    data = event.get("data", {})
                    if not isinstance(data, dict):
                        raise APIError("The API returned an unreadable stream event.")
                    if event_type == "sources":
                        sources = data.get("sources", [])
                        metadata.update(data)
                        evidence_loading_slot.empty()
                        evidence_content_slot.empty()
                        with evidence_content_slot.container():
                            render_sources(sources, metadata)
                    elif event_type == "token":
                        draft_answer += str(data.get("token", ""))
                        answer_slot.markdown(draft_answer)
                    elif event_type == "answer_replacement":
                        replacement_answer = str(data.get("answer", "")).strip() or None
                        if replacement_answer is not None:
                            answer_slot.markdown(replacement_answer)
                    elif event_type == "stage":
                        raw_name = str(data.get("name", "working")).replace("_", " ")
                        raw_status = str(data.get("status", "running")).replace("_", " ")
                        name = safe(raw_name.title())
                        status = safe(raw_status.title())
                        state_class = (
                            "is-complete"
                            if raw_status == "complete"
                            else "is-error"
                            if raw_status == "failed"
                            else "is-warning"
                            if raw_status
                            in {
                                "needs repair",
                                "needs review",
                                "unavailable",
                                "withheld",
                                "skipped",
                                "disabled",
                            }
                            else ""
                        )
                        dot_class = (
                            "offline"
                            if state_class == "is-error"
                            else "warning"
                            if state_class == "is-warning"
                            else "ready"
                        )
                        stage_slot.markdown(
                            f'<div class="pipeline-state {state_class}">'
                            f'<span class="status-dot {dot_class}"></span>'
                            f"<strong>{name}</strong><span>{status}</span></div>",
                            unsafe_allow_html=True,
                        )
                    elif event_type == "warning":
                        warnings.append(data.get("message", "The server reported a warning."))
                    elif event_type == "done":
                        done_received = True
                        final_metadata = data.get("metadata", data)
                        if isinstance(final_metadata, dict):
                            metadata.update(final_metadata)
                        final_answer = str(data.get("answer", "")).strip() or None
                    elif event_type == "error":
                        error_message = data.get("message", "Generation failed.")
                if error_message:
                    answer_slot.empty()
                    evidence_loading_slot.empty()
                    stage_slot.markdown(
                        '<div class="pipeline-state is-error"><span class="status-dot offline"></span>'
                        "<strong>Generation</strong><span>Failed</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.error(error_message, icon=None)
                elif not done_received:
                    raise APIError(
                        "The answer stream ended before final validation completed. "
                        "Retry the request."
                    )
                else:
                    answer = final_answer or replacement_answer or draft_answer.strip()
                    if not answer:
                        raise APIError("The API completed without returning a final answer.")
                    history.append(
                        {
                            "question": question.strip(),
                            "answer": answer,
                            "sources": sources,
                            "metadata": metadata,
                        }
                    )
                    history_committed = True
                    st.rerun()
            except APIError as exc:
                answer_slot.empty()
                evidence_loading_slot.empty()
                stage_slot.markdown(
                    '<div class="pipeline-state is-error"><span class="status-dot offline"></span>'
                    "<strong>Connection</strong><span>Interrupted</span></div>",
                    unsafe_allow_html=True,
                )
                st.error(str(exc), icon=None)
            if not history_committed:
                for warning in warnings:
                    st.warning(warning, icon=None)
    elif history:
        latest = history[-1]
        with answer_col, st.container(key="answer_sheet"):
            render_saved_answer(latest)
            if len(history) > 1:
                with st.expander(f"Earlier questions ({len(history) - 1})"):
                    for entry in reversed(history[:-1]):
                        render_saved_answer(entry)
        with evidence_col:
            with st.container(key="evidence_rail"):
                render_sources(latest.get("sources", []), latest.get("metadata"))
                if st.button("Clear conversation", use_container_width=True):
                    history.clear()
                    st.rerun()
    else:
        with answer_col, st.container(key="answer_sheet"):
            st.markdown(
                """
                <section class="empty-sheet">
                  <h3>Ask for a synthesis, not a guess</h3>
                  <p>The answer streams from the configured generation provider and is
                  constrained to retrieved abstracts. Numbered citations map directly
                  to the evidence ledger.</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
        with evidence_col:
            with st.container(key="empty_evidence_rail"):
                render_sources([])
                render_pipeline(health, label="How this answer is produced")


def main() -> None:
    configure_page()
    api_base_url = configured_api_base_url()
    backend_api_token = configured_backend_token()
    timeout_seconds = configured_api_request_timeout()
    client = get_client(api_base_url, backend_api_token, timeout_seconds)
    health = None
    health_error = None
    try:
        health = get_health(api_base_url, backend_api_token, timeout_seconds)
    except APIError as exc:
        health_error = str(exc)

    render_masthead(health, health_error)
    mode = st.radio(
        "Workspace mode",
        options=["Search", "Ask"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Search":
        render_search(client, health)
    else:
        render_ask(client, health)


if __name__ == "__main__":
    main()
