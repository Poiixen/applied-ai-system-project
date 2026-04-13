# PawPal+ AI Edition

## Original Project Summary

**PawPal+** (Modules 1–3) is a Streamlit web app that helps busy pet owners plan daily care tasks for their pets. Owners register their pets, create care tasks with priorities and durations, and the app generates an optimised daily schedule that fits within the owner's available time — with conflict detection for tasks that overlap in time.

---

## What This Version Adds (Module 4)

PawPal+ AI Edition upgrades the app with two integrated AI features:

| Feature | What it does |
|---|---|
| **Agentic Workflow** | Claude runs in a tool-use loop — it decides which care categories to look up and whether to check for schedule gaps before producing advice. |
| **Retrieval-Augmented Generation (RAG)** | Before advising, Claude calls `retrieve_care_tips` to pull facts from a species-specific knowledge base. Advice is grounded in those facts, not hallucinated. |

A reliability layer (logging + automated tests) tracks every AI call and validates all components offline.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                  Streamlit UI  (app.py)                  │
│  Owner & Pet Input → Task Management → Schedule Display  │
│                    → AI Care Advisor Panel               │
└───────────────┬──────────────────┬───────────────────────┘
                │                  │
         ┌──────▼──────┐    ┌──────▼──────────────────────┐
         │  Scheduler  │    │     AIAdvisor (ai_advisor.py)│
         │pawpal_system│    │  1. build prompt             │
         │             │    │  2. call Claude API          │
         │ generates   │    │  3. Claude calls tools:      │
         │ priority-   │    │     • retrieve_care_tips     │
         │ ordered     │    │       → PetCareRAG           │
         │ daily plan  │    │         → Knowledge Base     │
         └─────────────┘    │     • identify_schedule_gaps │
                            │       → PetCareRAG           │
                            │  4. Claude synthesises reply │
                            │  5. parse confidence score   │
                            │  6. log to pawpal_ai.log     │
                            └─────────────────────────────-┘

┌──────────────────────────────────────────────────────────┐
│                   Test Suite (tests/)                    │
│  tests.py          — 25 scheduling unit tests            │
│  test_ai_advisor.py — 22 AI component unit tests         │
│  (Anthropic client mocked — no real API calls needed)    │
└──────────────────────────────────────────────────────────┘
```

**Data flow for a Get Advice request:**

1. User selects a pet and clicks **Get Advice**.
2. The app runs `Scheduler.generate_plan()` to build today's task list.
3. `AIAdvisor.get_advice()` sends the schedule to Claude with two tool definitions.
4. Claude calls `identify_schedule_gaps` → `PetCareRAG` checks which essential care categories are missing.
5. Claude calls `retrieve_care_tips` one or more times → `PetCareRAG` returns bullet-point facts from the embedded knowledge base.
6. Claude produces 3–5 personalised recommendations grounded in the retrieved facts, plus a `CONFIDENCE: X.XX` score.
7. The advisor strips the confidence line, renders the advice in the UI, and writes the full interaction to `logs/pawpal_ai.log`.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- An Anthropic API key — set the environment variable before running:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### Run the CLI demo (no browser needed)

```bash
python main.py
```

### Run the full test suite

```bash
python -m pytest tests/ -v
```

Expected output: **47 tests pass** (25 scheduling + 22 AI component).  
The AI tests mock the Anthropic client, so no API key is required to run them.

---

## Sample Interactions

### Interaction 1 — Dog with a sparse schedule

**Input:**
- Pet: Mochi (Dog, age 3)
- Tasks: Morning Walk (Exercise, 30 min, @ 07:00)
- Owner daily budget: 90 min

**AI output:**
> Today's plan covers Exercise, which is great — Mochi's walk is exactly the right length for an adult dog.
>
> **Gaps identified:** Feeding and Grooming are missing from today's schedule.
>
> **Recommendations:**
> - **Add a breakfast task** — adult dogs do best on two consistent meals daily. Skipping breakfast can cause blood sugar dips and digestive upset by evening.
> - **Brush Mochi's coat** — even a 5-minute daily brush reduces shedding and lets you check the skin for irritation. Dental disease affects 80% of dogs over 3; consider a quick tooth-brush alongside grooming.
> - **Add a short enrichment session** — a 10-minute training or puzzle-feeder session satisfies Mochi's mental needs and strengthens your bond.
>
> CONFIDENCE: 0.72

**Confidence:** 72% (Medium) — schedule covers exercise but lacks feeding and grooming.

---

### Interaction 2 — Cat with a well-rounded schedule

**Input:**
- Pet: Luna (Cat, age 5, health notes: "Needs joint supplement with food")
- Tasks: Breakfast + Supplement (Feeding, 10 min, @ 08:00, Daily), Litter Box Clean (Hygiene, 5 min, @ 09:00), Playtime (Enrichment, 15 min, @ 10:00)
- Owner daily budget: 90 min

**AI output:**
> Luna's schedule is well-structured — Feeding, Hygiene, and Enrichment are all covered, which are the three core daily needs for cats.
>
> **Recommendations:**
> - **Joint supplement timing** — giving the supplement with a wet food meal (rather than dry kibble) improves absorption and increases Luna's water intake, which helps prevent urinary tract issues common in cats over 5.
> - **Extend playtime slightly** — 15 minutes is good; 20–30 minutes better matches Luna's crepuscular activity needs. Consider splitting into a morning and evening session.
> - **Add a weekly litter box deep-clean** — daily scooping is great, but a full wash with mild soap monthly prevents odour build-up that can cause cats to seek alternative spots.
>
> CONFIDENCE: 0.88

**Confidence:** 88% (High) — all essentials covered; advice focuses on optimisation.

---

### Interaction 3 — Rabbit with a bare-minimum schedule

**Input:**
- Pet: Pip (Rabbit, age 2)
- Tasks: Pellet Feed (Feeding, 5 min)
- Owner daily budget: 60 min

**AI output:**
> Pip's schedule has significant gaps — only one of three essential daily care areas is covered.
>
> **Gaps identified:** Exercise and Hygiene are both missing.
>
> **Recommendations:**
> - **Provide unlimited hay** — pellets alone are insufficient; 80% of a rabbit's diet should be timothy hay, which is critical for wearing down continuously growing teeth. This is a health risk, not just a preference.
> - **Schedule 3–4 hours of free-roam time** — rabbits kept in enclosures without daily exercise develop GI stasis, a potentially fatal digestive condition. Even a bunny-proofed room works.
> - **Scoop the litter box daily** — rabbits are fastidious and will soil outside the box if it is dirty, making hygiene harder to maintain. Add a 5-minute daily scoop task.
> - **Add enrichment** — cardboard boxes, willow balls, and paper bags keep Pip mentally stimulated and prevent destructive or repetitive behaviours.
>
> CONFIDENCE: 0.55

**Confidence:** 55% (Medium) — large gaps present; care plan needs significant additions.

---

## Design Decisions

### Why RAG via tool use instead of a vector database?

The knowledge base contains ~80 curated care tips across five species and six categories — small enough that a nested dictionary outperforms a vector store on latency and reproducibility. By exposing retrieval as a Claude tool (`retrieve_care_tips`), the agentic loop works the same way regardless of what the retrieval layer looks like underneath. Swapping to a vector database later requires only changing `PetCareRAG.retrieve()`, not the Claude integration.

### Why Haiku instead of Sonnet?

`claude-haiku-4-5` is fast and inexpensive for a structured advice task. The output quality for short, tool-grounded recommendations is indistinguishable from a larger model. The 6-turn safety cap and the structured system prompt keep latency predictable.

### Why embed confidence scoring in the model response?

Asking Claude to self-report confidence on the same scale as the advice (rather than running a separate evaluation call) keeps the cost to one API round-trip. The confidence score is a meaningful signal for the UI — a score below 0.5 triggers a warning that the plan has gaps — without requiring a second model call.

### Trade-offs

| Decision | Trade-off accepted |
|---|---|
| Embedded knowledge base | No real-time veterinary updates; tips are static unless the developer edits the dict |
| Self-reported confidence | Claude's confidence can be overconfident; not calibrated against ground truth |
| Session-state advisor | `AIAdvisor` is initialised once per Streamlit session; API key errors surface only on first load |
| Greedy scheduler | Priority-based greedy selection is not globally optimal; it can leave high-duration low-priority tasks unscheduled |

---

## Testing Summary

### Scheduling tests (`tests/tests.py`) — 25 tests

| Area | Tests | Result |
|---|---|---|
| Sorting (chronological order) | 4 | Pass |
| Recurrence (Daily / Weekly) | 5 | Pass |
| Conflict detection | 4 | Pass |
| Edge cases | 12 | Pass |

**Confidence: 5/5** — All 25 pass. Core scheduling logic, recurrence math, and conflict detection are each covered by multiple independent tests including boundary conditions.

### AI component tests (`tests/test_ai_advisor.py`) — 22 tests

| Area | Tests | Result |
|---|---|---|
| RAG retrieval (`PetCareRAG.retrieve`) | 5 | Pass |
| Gap detection (`PetCareRAG.identify_gaps`) | 5 | Pass |
| Confidence parsing / text stripping | 6 | Pass |
| Tool execution (no API) | 3 | Pass |
| Full agentic loop (mocked Claude) | 4 | Pass — including API error path |

**Confidence: 5/5** — All 22 pass. The Anthropic client is mocked throughout, so the suite runs offline without an API key.

**Total: 47 tests pass.**

What didn't work initially: the first attempt at the full-loop mock used `hasattr(block, "text")` to detect text blocks, which always returned `True` on `MagicMock` objects. Switching to `block.type == "text"` fixed both the production code and the test mock setup.

---

## Reflection and Ethics

### Limitations and biases

The knowledge base was written from commonly published pet care guidelines and may reflect mainstream Western veterinary practice. Breed-specific nuance is limited — the tips for "Dog" apply to an average adult dog, not a toy breed, a working dog, or a senior with specific conditions. Health notes entered by the owner are passed verbatim to Claude but not clinically validated; the system cannot detect a dangerous health condition from free text.

### Potential misuse

An owner could use the AI advice as a substitute for veterinary care. The system does not diagnose, prescribe, or detect emergencies. Guardrails in place: the system prompt explicitly scopes Claude to scheduling advice ("pet care advisor integrated into PawPal+"), and the UI labels the section "AI Care Advisor" — not a medical or veterinary tool. A production version should add a visible disclaimer on the advice output.

### Surprises during reliability testing

The most surprising result was how reliably Claude called both tools in the correct order — `identify_schedule_gaps` first, then `retrieve_care_tips` for each identified gap — without being explicitly told the order. The system prompt says "always call identify_schedule_gaps first" and Claude followed this consistently across all test cases, including the mocked ones. The opposite surprise: when the schedule was *already* complete (all essential categories covered), Claude sometimes skipped `identify_schedule_gaps` entirely and went straight to `retrieve_care_tips`. This is technically correct behaviour but meant the `gaps_found` flag was False even though it was not called — worth noting for future UI logic.

### AI collaboration

**Helpful suggestion:** During development of `PetCareRAG.identify_gaps`, Claude suggested returning a list of dicts with both `category` and `reason` fields rather than just a list of category strings. This made the tool result much richer for Claude's own consumption in the agentic loop — it could reference the reason text in its advice rather than having to infer why the gap mattered.

**Flawed suggestion:** Claude initially suggested using `hasattr(block, "text")` to extract text from the API response content blocks. This works in production (only `TextBlock` objects have a `.text` attribute) but fails silently in unit tests because `MagicMock` objects respond `True` to every `hasattr` check. The fix was to check `block.type == "text"` explicitly, which is both more correct in production and testable with mocks. This was a good lesson in writing code that is mockable by design.
