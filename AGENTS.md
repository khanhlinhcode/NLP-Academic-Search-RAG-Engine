# Project Agent Skills & Instructions

This repository is equipped with the following custom AI Agent Skills:

## Registered Skills

1. **`ponytail`** (`.agents/skills/ponytail/SKILL.md`):
   - **Trigger:** Enforce pragmatic code design, YAGNI, standard library usage, token efficiency, anti-over-engineering.
   - **Usage:** Call or refer to `$ponytail` / `ponytail` to audit and keep code minimal.

2. **`nlp-model-design`** (`.agents/skills/nlp-model-design/SKILL.md`):
   - **Trigger:** NLP architecture design, Hybrid BM25 + Vector RRF search, 512MB RAM constraints, Model Cards, IR evaluation metrics.
   - **Usage:** Call or refer to `$nlp-model-design` / `nlp-model-design` for NLP pipeline engineering.

3. **`semantically-verified-rag`** (`.agents/skills/semantically-verified-rag/SKILL.md`):
   - **Trigger:** Provider-neutral 2-layer semantic evidence verification, exact substring quote checking, Groq JSON Schema structured outputs, 1-pass repair loop, fail-closed safety.
   - **Usage:** Call or refer to `$semantically-verified-rag` / `semantically-verified-rag` for grounding verification tasks.
