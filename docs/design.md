# Design System

<!-- impeccable:design-schema 1 -->

## Direction

The interface follows **Premium Scholarly Editorial — Private Research Library**. It behaves
like an annotated academic proof sheet: a working document where retrieval evidence, ranking,
citations, and model state are visible at a glance. It should feel authored for research rather
than dressed as a generic AI dashboard.

## Visual World

- Warm ivory paper (`#F3EFE6`), near-black ink (`#1D1D1A`), graphite rules, and a restrained
  oxblood accent (`#762F35`). Primary surfaces use `#FBF9F4`; secondary surfaces use `#EAE4D9`.
- The accent marks actions, active citations, and consequential system states; it is not decorative fill.
- Information is organized through editorial columns, folio-like labels, marginal notes, and typographic hierarchy rather than floating card chrome.
- Surfaces are mostly flat. Borders and spacing establish structure; shadows are avoided except where native Streamlit overlays require separation.
- Gradients, neon glow, glassmorphism, decorative blur, and generic AI imagery do not belong to
  this visual world.

## Typography

- Display and paper titles use the bundled IM Fell French Canon with a sturdy system serif
  fallback for unsupported glyphs.
- Controls, metadata, scores, and system labels use a compact sans/monospace voice.
- IBM Plex Mono is reserved for identifiers, measurements, status metadata, and rank notation.
- Fonts are bundled with the UI and embedded locally; the production surface does not depend on
  a third-party font CDN.
- Body copy remains highly legible at a moderate measure; abstracts should not exceed a comfortable reading width.
- Sentence case is the default. Short instrument labels may use tracked uppercase.

## Composition

- A slim masthead carries product identity and live system status.
- The first viewport puts the query instrument first and immediately reveals how the engine will process it.
- Search results read as ranked entries in a proceedings index: rank, title, metadata, score, and an expandable abstract.
- RAG answers use a central reading column with citations and source notes in a clearly adjacent evidence region.
- Advanced controls stay in a narrow rail or disclosure region and collapse cleanly on small screens.
- Retrieval stages and readiness checks live inside a collapsed explanatory disclosure so that
  they remain inspectable without competing with search results or answers.

## Components and States

- Primary actions use solid oxblood with strong contrast; secondary controls use paper/ink outlines.
- Tabs behave as a two-part mode switch, not as detached navigation pills.
- Status language is explicit: connected, API unavailable, Ollama ready, model missing, retrieving, reranking, generating.
- Empty states teach with concrete example queries.
- Errors state what failed, what remains usable, and the next local command when known.
- Loading states preserve layout and identify the active pipeline stage.

## Interaction and Motion

- The signature interaction is evidence accumulation: as generation streams, the answer grows while numbered source notes remain stable and inspectable.
- Result entries reveal abstracts in place without navigating away.
- Motion is limited to short state transitions and a subtle streaming caret; `prefers-reduced-motion` removes nonessential animation.

## Responsive Behavior

- Wide screens use a control rail plus a primary reading area.
- Narrow screens linearize controls before content, keep actions full-width, and avoid horizontal scrolling.
- Metadata wraps by meaning, not by forcing fixed columns.

## Accessibility

- Maintain visible keyboard focus and semantic labels.
- Never encode health, method, or score using color alone.
- Ensure the oxblood/ink/paper combinations meet WCAG AA for the text sizes used.
- Do not rasterize interface text or controls.
