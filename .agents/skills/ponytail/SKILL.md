---
name: ponytail
description: Skill designed to prevent over-engineering in AI coding agents. Enforces strict simplicity, code reuse, YAGNI, standard library prioritization, minimal dependencies, and token-efficient solutions.
version: 1.0.0
---

# Ponytail Skill: Pragmatic & Anti-Over-Engineering Guidelines

The **Ponytail** skill forces AI agents to think like the most pragmatic, senior developer in the room. The objective is to produce lean, maintainable, robust code with zero bloated abstractions, unnecessary wrapper layers, or redundant dependencies.

---

## 1. Core Philosophy: The Decision Ladder

Before writing any new function, class, file, or dependency, the agent **MUST** pass through the **7-Step Decision Ladder**:

```
[1] Does it need to exist? (YAGNI)
   │   └─ YES ──> [2] Is it already in the codebase? (REUSE)
   │                 │   └─ NO ──> [3] Does the standard library do it? (STDLIB)
   │                 │               │   └─ NO ──> [4] Is there a native platform/framework feature?
   │                 │               │               │   └─ NO ──> [5] Is there an installed dependency?
   │                 │               │               │               │   └─ NO ──> [6] Can it be written in a single clean expression?
   │                 │               │               │               │               │   └─ NO ──> [7] Write the MINIMUM working code.
```

1. **YAGNI (You Aren't Gonna Need It):** If a feature, helper function, or edge-case handler is not explicitly required by the specification or test case, **DO NOT WRITE IT**.
2. **Reuse Codebase First:** Check existing utilities (`src/`, `utils/`, `helpers/`) before writing a new helper function.
3. **Standard Library First:** Prioritize built-in Python modules (`pathlib`, `json`, `dataclasses`, `typing`, `asyncio`, `urllib`, `re`, `hashlib`, etc.) over third-party dependencies.
4. **Native Platform/Framework Feature:** Use native FastAPI, Pydantic, Streamlit, or PyTorch capabilities instead of custom hacks.
5. **Leverage Installed Dependencies:** Use already imported packages before introducing new `pip` / `uv` dependencies.
6. **Prefer Simple Expressions:** Prefer concise list comprehensions, built-in functions, or inline ternary expressions over multiline loop abstractions if clear.
7. **Minimum Viable Code:** Write the absolute smallest implementation that satisfies test cases and safety constraints.

---

## 2. Mandatory Rules

### Rule A: No Premature Abstraction
- Do **NOT** create abstract base classes, generic interfaces, or factory patterns unless there are at least **two concrete, distinct implementations** actively required in the codebase.
- Avoid multi-level inheritance hierarchies. Prefer simple composition or typed functions.

### Rule B: Minimal API Surface & Zero Token Bloat
- Keep function signatures short and typed with Pydantic or standard Python type hints.
- Do not add "just-in-case" configuration options, unused default parameters, or placeholder extension points.
- Keep comments concise and focused on non-obvious design rationale or security constraints. Do not restate what code line-by-line does.

### Rule C: Refactoring Constraints
- Never refactor working code outside the requested target scope.
- Preserve existing APIs, field names, and external contracts unless explicitly tasked with breaking changes.
- Never add heavy dependencies (e.g., PyTorch, Transformers, spaCy) into lightweight service containers (e.g., Render 512MB RAM budget) when HTTP calls or lightweight protocols suffice.

### Rule D: Strict Error Handling & Fail-Closed Behavior
- Prefer simple typed exceptions over generic `catch-all` wrappers.
- When validation fails in high-integrity components (e.g., RAG citation verification), fail closed gracefully with clear refusal messages rather than fabricating data.

---

## 3. Code Audit & Self-Checklist

Before finalizing any task, review the code against this audit checklist:

- [ ] **Did I add any file, class, or function that could be deleted without breaking functionality or tests?**
- [ ] **Did I re-implement anything that standard python modules (`pathlib`, `json`, `functools`) already handle?**
- [ ] **Are all new functions actively called in the execution path or tests?**
- [ ] **Is the solution free of extra architectural layers (e.g., unnecessary adapters, redundant DTOs)?**
- [ ] **Is the RAM footprint minimal and compatible with low-spec deployment targets?**
