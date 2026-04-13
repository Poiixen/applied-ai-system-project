"""
ai_advisor.py — Agentic AI Care Advisor for PawPal+

Architecture:
  PetCareRAG  — retrieves species/category tips from the embedded knowledge base
  AIAdvisor   — runs an agentic Claude loop: Claude calls tools (retrieve_care_tips,
                 identify_schedule_gaps) to ground its advice in factual guidelines,
                 then synthesises recommendations with a self-reported confidence score.

All interactions are logged to logs/pawpal_ai.log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Logging setup — writes to logs/pawpal_ai.log AND the console
# ---------------------------------------------------------------------------

_logs_dir = Path("logs")
_logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(_logs_dir / "pawpal_ai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pawpal_ai")

# ---------------------------------------------------------------------------
# Knowledge base (RAG source) — species × care-category → list of fact strings
# ---------------------------------------------------------------------------

CARE_KNOWLEDGE_BASE: dict[str, dict[str, list[str]]] = {
    "Dog": {
        "Exercise": [
            "Adult dogs generally need 30–60 minutes of vigorous exercise daily; puppies and seniors need gentler, shorter sessions.",
            "Mental stimulation (puzzle feeders, nose work) reduces destructive behaviour and supplements physical walks.",
            "High-energy breeds (Border Collie, Husky, Labrador) may need 90+ minutes of activity daily.",
            "Avoid strenuous exercise 1 hour before and after meals to reduce bloat risk in deep-chested breeds.",
        ],
        "Feeding": [
            "Adult dogs do best on two meals per day at consistent times to regulate digestion and energy.",
            "Fresh water should always be available; dehydration causes lethargy and kidney stress.",
            "Avoid feeding table scraps — onions, grapes, raisins, chocolate, and xylitol are toxic to dogs.",
            "Portion control is critical; obesity is the leading preventable health problem in dogs.",
        ],
        "Grooming": [
            "Brush coats 2–3× weekly (daily for long/double coats) to prevent mats and distribute skin oils.",
            "Teeth should be brushed 2–3× weekly minimum; dental disease affects 80% of dogs over age 3.",
            "Nails should be trimmed every 3–4 weeks; overgrown nails alter gait and cause joint stress.",
            "Check ears weekly for redness, odour, or discharge — floppy-eared breeds are prone to infection.",
        ],
        "Enrichment": [
            "Dogs need daily mental challenges: training sessions, interactive toys, or new environments to sniff.",
            "Socialisation with other dogs and people prevents anxiety and aggression.",
            "Learning new commands or tricks keeps the brain active and strengthens the human-dog bond.",
        ],
        "Hygiene": [
            "Bathe dogs every 4–6 weeks or when dirty; over-bathing strips natural skin oils.",
            "Wipe paws after outdoor walks, especially in winter (salt) or after pesticide-treated areas.",
            "Anal glands may need monthly expression in some breeds — a vet or groomer can advise.",
        ],
        "Medication": [
            "Administer medications at the same time each day to maintain consistent blood levels.",
            "Never skip doses of prescription medication without consulting a vet.",
            "Many common human medications (ibuprofen, acetaminophen) are toxic to dogs — never share.",
        ],
    },
    "Cat": {
        "Exercise": [
            "Cats need 20–30 minutes of active interactive play daily using wands, lasers, or puzzle feeders.",
            "Indoor cats especially benefit from vertical space (cat trees, shelves) to satisfy climbing instincts.",
            "Two shorter play sessions (morning/evening) match a cat's natural crepuscular activity pattern.",
        ],
        "Feeding": [
            "Cats are obligate carnivores and require animal-based protein; grain-heavy diets can cause obesity.",
            "Multiple small meals (3–4× daily) better match feline metabolism than one large meal.",
            "Wet food helps prevent urinary tract issues by increasing water intake.",
            "Fresh water should always be available; many cats prefer running water — consider a fountain.",
        ],
        "Grooming": [
            "Short-haired cats self-groom adequately; brush weekly to reduce shedding and hairballs.",
            "Long-haired cats (Persians, Maine Coons) need daily brushing to prevent painful mats.",
            "Dental disease affects 70% of cats over 3; aim for weekly tooth brushing.",
            "Trim nails every 2–3 weeks to prevent overgrowth and furniture damage.",
        ],
        "Enrichment": [
            "Cats need environmental complexity: scratching posts, window perches, and hiding spots.",
            "Puzzle feeders slow eating and provide mental stimulation for indoor cats.",
            "Rotate toys weekly to maintain novelty; cats habituate to the same toys quickly.",
        ],
        "Hygiene": [
            "Scoop litter boxes at least once daily; cats may refuse to use dirty boxes and eliminate elsewhere.",
            "Use one litter box per cat plus one extra to prevent territorial disputes.",
            "Clean litter boxes monthly with mild soap; strong scents deter cats.",
        ],
        "Medication": [
            "Cats are highly sensitive to many drugs — never give canine medications to a cat.",
            "Mix medications with a small amount of wet food if the cat resists direct administration.",
            "Maintain consistent medication timing; stress from irregular schedules affects feline health.",
        ],
    },
    "Rabbit": {
        "Exercise": [
            "Rabbits need at least 3–4 hours of free-roaming exercise daily outside their enclosure.",
            "Provide tunnels, platforms, and digging boxes to satisfy natural burrowing instincts.",
            "Lack of exercise causes GI stasis, obesity, and muscle wasting in rabbits.",
        ],
        "Feeding": [
            "80% of a rabbit's diet should be unlimited timothy hay — it wears down continuously growing teeth.",
            "Fresh leafy greens (romaine, kale, cilantro) provide vitamins; avoid iceberg lettuce.",
            "Limit pellets to 1/4 cup per 5 lbs of body weight; excess causes obesity.",
            "Fresh water is essential; use both a bowl and bottle as backup.",
        ],
        "Grooming": [
            "Brush rabbits 2–3× weekly (daily during moults) to prevent ingesting hair, which causes GI blockage.",
            "Never bathe rabbits in water — it causes extreme stress and hypothermia; spot-clean instead.",
            "Trim nails every 4–6 weeks; overgrown nails catch on surfaces and cause painful breaks.",
        ],
        "Enrichment": [
            "Rabbits are social and need daily interaction; bonded pairs thrive more than single rabbits.",
            "Provide cardboard boxes, willow balls, and paper bags for safe chewing and exploration.",
            "Rabbits learn names and can be trained with positive reinforcement using small treats.",
        ],
        "Hygiene": [
            "Spot-clean the litter box daily and deep-clean weekly; rabbits are fastidious and prefer a clean toilet area.",
            "Check dewlap folds on female rabbits for moisture or skin irritation weekly.",
            "Monitor droppings daily — change in size or consistency signals dietary or health issues.",
        ],
        "Medication": [
            "Rabbits require species-specific medications; never use dog or cat products on rabbits.",
            "Administer medications at consistent times daily and hide them in a small piece of banana if needed.",
        ],
    },
    "Bird": {
        "Exercise": [
            "Birds need at least 2–3 hours of out-of-cage time daily for flight and social interaction.",
            "Flight is critical for respiratory and cardiovascular health; clipped birds still need active out-of-cage time.",
            "Foraging activities (hiding food, shreddable toys) satisfy natural behaviour and provide exercise.",
        ],
        "Feeding": [
            "Fresh vegetables, leafy greens, and limited fruit should make up 50–70% of a parrot's diet.",
            "High-quality pellets (not seeds alone) should form the dietary base for most companion parrots.",
            "Avocado, onion, chocolate, and caffeine are toxic to birds; ensure no exposure.",
            "Fresh water should be changed twice daily minimum; birds are sensitive to bacterial contamination.",
        ],
        "Grooming": [
            "Nail trimming every 4–6 weeks prevents overgrowth and perch-related foot injuries.",
            "Offer a shallow water dish or misting 2–3× weekly; most birds enjoy bathing.",
            "Beak trimming is rarely needed if perch surfaces vary in texture; consult a vet if overgrown.",
        ],
        "Enrichment": [
            "Rotate 10–15 toys, cycling new ones in weekly to prevent boredom and feather-destructive behaviour.",
            "Social interaction is critical; birds bonded to owners need 2+ hours of direct interaction daily.",
            "Teaching tricks and words provides mental stimulation and prevents screaming behaviours.",
        ],
        "Hygiene": [
            "Clean food and water dishes daily with hot soapy water — bacteria grow rapidly in bird environments.",
            "Disinfect the cage weekly with bird-safe cleaner; remove droppings from perches daily.",
            "Change cage liner (paper) daily to minimise ammonia build-up from droppings.",
        ],
        "Medication": [
            "Birds require avian-specific medications; consult an avian vet rather than using dog/cat products.",
            "Mix liquid medications into a small amount of favourite food if the bird resists direct dosing.",
        ],
    },
    "Other": {
        "Exercise": [
            "Research species-specific activity requirements; exotic pets vary widely in exercise needs.",
            "Provide an enriched environment with hiding spots, climbing structures, and natural behaviours.",
        ],
        "Feeding": [
            "Research species-appropriate diet; exotic pets have highly specific nutritional requirements.",
            "Fresh water should always be available in a form appropriate for the species.",
        ],
        "Grooming": [
            "Research grooming requirements specific to the species; over- or under-grooming can cause harm.",
        ],
        "Enrichment": [
            "Environmental enrichment should mimic natural habitat as closely as possible.",
            "Social needs vary widely; research whether the species is solitary or social.",
        ],
        "Hygiene": [
            "Maintain habitat cleanliness on a schedule appropriate to the species' sensitivities.",
        ],
        "Medication": [
            "Exotic animals require species-specific medications; always consult an exotic vet.",
        ],
    },
}

# Core care categories that should appear in every pet's daily schedule
ESSENTIAL_CATEGORIES: dict[str, list[str]] = {
    "Dog":    ["Exercise", "Feeding", "Grooming"],
    "Cat":    ["Feeding", "Enrichment", "Hygiene"],
    "Rabbit": ["Feeding", "Exercise", "Hygiene"],
    "Bird":   ["Feeding", "Enrichment", "Hygiene"],
    "Other":  ["Feeding"],
}

# ---------------------------------------------------------------------------
# Tool definitions sent to Claude
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "retrieve_care_tips",
        "description": (
            "Retrieve evidence-based pet care tips from the knowledge base for a specific species "
            "and care category. Use this to ground advice in factual guidelines before making "
            "recommendations. Always call this before advising on a category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "The pet's species (Dog, Cat, Rabbit, Bird, or Other).",
                    "enum": ["Dog", "Cat", "Rabbit", "Bird", "Other"],
                },
                "category": {
                    "type": "string",
                    "description": "The care category to look up.",
                    "enum": ["Exercise", "Feeding", "Grooming", "Enrichment", "Hygiene", "Medication"],
                },
            },
            "required": ["species", "category"],
        },
    },
    {
        "name": "identify_schedule_gaps",
        "description": (
            "Analyse the owner's current scheduled task categories and identify which core care "
            "areas are missing or underrepresented for the given species. Returns a list of gap "
            "objects with a category name and a brief explanation of why it matters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "The pet's species.",
                    "enum": ["Dog", "Cat", "Rabbit", "Bird", "Other"],
                },
                "scheduled_categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of care categories already present in today's schedule.",
                },
            },
            "required": ["species", "scheduled_categories"],
        },
    },
]


# ---------------------------------------------------------------------------
# PetCareRAG — retrieval layer
# ---------------------------------------------------------------------------

class PetCareRAG:
    """Retrieves relevant care tips from the embedded knowledge base."""

    def retrieve(self, species: str, category: str) -> list[str]:
        """Return care tips for the given species and category."""
        species_db = CARE_KNOWLEDGE_BASE.get(species, CARE_KNOWLEDGE_BASE["Other"])
        tips = species_db.get(category)
        if tips:
            return tips
        return [
            f"No specific tips found for {species} — {category}. "
            "Consult a veterinarian for species-appropriate guidance."
        ]

    def identify_gaps(self, species: str, scheduled_categories: list[str]) -> list[dict[str, str]]:
        """Return gap dicts for essential care categories missing from the schedule."""
        essential = ESSENTIAL_CATEGORIES.get(species, ["Feeding"])
        scheduled_set = set(scheduled_categories)
        return [
            {
                "category": cat,
                "reason": (
                    f"{cat} is an essential daily care need for {species}s "
                    "but is not represented in today's scheduled tasks."
                ),
            }
            for cat in essential
            if cat not in scheduled_set
        ]


# ---------------------------------------------------------------------------
# AIAdvisor — agentic Claude loop
# ---------------------------------------------------------------------------

class AIAdvisor:
    """
    Agentic AI advisor that gives Claude a knowledge-retrieval tool and a gap-analysis
    tool, then lets Claude decide which information to pull before producing advice.

    Workflow per call to get_advice():
      1. Build a prompt with the pet's info and today's scheduled tasks.
      2. Call Claude with both tool definitions.
      3. If Claude returns tool_use blocks, execute each tool and feed results back.
      4. Repeat until Claude returns end_turn (max 6 turns as a safety guard).
      5. Parse the confidence score Claude embeds in its final message.
      6. Log everything and return a result dict.
    """

    _MAX_TURNS = 6
    _MODEL = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.rag = PetCareRAG()
        logger.info("AIAdvisor initialised with model=%s", self._MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_advice(self, pet: Any, scheduled_tasks: list[Any], owner: Any) -> dict[str, Any]:
        """
        Run the agentic advice loop for the given pet and schedule.

        Returns a dict with keys:
          advice           — formatted recommendation string (CONFIDENCE line stripped)
          confidence       — float in [0, 1]
          tools_used       — list of tool names Claude called
          gaps_found       — True if identify_schedule_gaps was called
          elapsed_seconds  — wall-clock time for the full call
          error            — present only if an API error occurred
        """
        start = datetime.now()
        logger.info(
            "Advice requested | pet=%s species=%s tasks=%d owner_budget=%d min",
            pet.name, pet.species, len(scheduled_tasks), owner.available_minutes_per_day,
        )

        messages = self._build_initial_messages(pet, scheduled_tasks, owner)
        tools_used: list[str] = []
        final_text = ""

        try:
            for turn in range(self._MAX_TURNS):
                response = self.client.messages.create(
                    model=self._MODEL,
                    max_tokens=1500,
                    system=self._system_prompt(),
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
                logger.debug("Turn %d | stop_reason=%s", turn + 1, response.stop_reason)

                if response.stop_reason == "end_turn":
                    final_text = self._extract_text(response)
                    break

                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            tools_used.append(block.name)
                            result_json = self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_json,
                            })
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    logger.warning("Unexpected stop_reason=%s", response.stop_reason)
                    final_text = self._extract_text(response)
                    break
            else:
                logger.warning("Max turns (%d) reached without end_turn", self._MAX_TURNS)
                final_text = (
                    "Advice generation reached the iteration limit. "
                    "Try again or simplify the schedule."
                )

        except anthropic.APIError as exc:
            logger.error("Anthropic API error | pet=%s | %s: %s", pet.name, type(exc).__name__, exc)
            return {
                "advice": f"AI advice is currently unavailable ({type(exc).__name__}).",
                "confidence": 0.0,
                "tools_used": tools_used,
                "gaps_found": False,
                "elapsed_seconds": (datetime.now() - start).total_seconds(),
                "error": str(exc),
            }

        confidence = self._parse_confidence(final_text)
        advice = self._strip_confidence_line(final_text)
        elapsed = (datetime.now() - start).total_seconds()

        logger.info(
            "Advice complete | pet=%s tools=%s confidence=%.2f elapsed=%.2fs",
            pet.name, tools_used, confidence, elapsed,
        )

        return {
            "advice": advice,
            "confidence": confidence,
            "tools_used": tools_used,
            "gaps_found": "identify_schedule_gaps" in tools_used,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a knowledgeable pet care advisor integrated into PawPal+, a daily pet care "
            "planner. Your job is to analyse a pet owner's scheduled care tasks and provide "
            "personalised, evidence-based recommendations.\n\n"
            "Rules:\n"
            "- Always call identify_schedule_gaps first to check for missing care areas.\n"
            "- Call retrieve_care_tips for each category you plan to advise on.\n"
            "- Ground every recommendation in the retrieved tips — do not rely solely on prior knowledge.\n"
            "- Be warm, specific, and concise (3–5 bullet points max).\n"
            "- End your final message with exactly this line: CONFIDENCE: X.XX\n"
            "  where X.XX is your confidence (0.00–1.00) that today's plan covers the pet's needs."
        )

    @staticmethod
    def _build_initial_messages(pet: Any, scheduled_tasks: list[Any], owner: Any) -> list[dict]:
        task_lines = "\n".join(
            f"  - {t.name} ({t.category}, {t.duration_minutes} min, priority {t.priority}/5"
            + (f", @ {t.scheduled_time}" if t.scheduled_time else "")
            + ")"
            for t in scheduled_tasks
        ) or "  (no tasks fit within today's time budget)"

        content = (
            f"Please analyse today's care plan for **{pet.name}** "
            f"({pet.species}, age {pet.age_years}).\n\n"
            f"Owner: {owner.name} | Daily time budget: {owner.available_minutes_per_day} min\n"
        )
        if pet.health_notes:
            content += f"Health notes: {pet.health_notes}\n"
        content += f"\nScheduled tasks today:\n{task_lines}\n\n"
        content += (
            "Please:\n"
            "1. Check for care gaps using identify_schedule_gaps.\n"
            "2. Retrieve tips for the most important care categories.\n"
            "3. Give 3–5 specific, actionable recommendations based on the retrieved tips.\n"
            "4. End with: CONFIDENCE: X.XX"
        )
        return [{"role": "user", "content": content}]

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return a JSON string result."""
        if name == "retrieve_care_tips":
            tips = self.rag.retrieve(tool_input["species"], tool_input["category"])
            logger.info(
                "RAG | species=%s category=%s -> %d tips",
                tool_input["species"], tool_input["category"], len(tips),
            )
            return json.dumps({"tips": tips, "count": len(tips)})

        if name == "identify_schedule_gaps":
            gaps = self.rag.identify_gaps(
                tool_input["species"], tool_input["scheduled_categories"]
            )
            logger.info(
                "Gap analysis | species=%s scheduled=%s -> %d gaps",
                tool_input["species"], tool_input["scheduled_categories"], len(gaps),
            )
            return json.dumps({"gaps": gaps, "gap_count": len(gaps)})

        logger.warning("Unknown tool requested: %s", name)
        return json.dumps({"error": f"Unknown tool: {name}"})

    @staticmethod
    def _extract_text(response: Any) -> str:
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    @staticmethod
    def _parse_confidence(text: str) -> float:
        """Extract the CONFIDENCE: X.XX score from the advice text."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("CONFIDENCE:"):
                try:
                    score = float(stripped.split(":", 1)[1].strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass
        return 0.75  # default if Claude omits the line

    @staticmethod
    def _strip_confidence_line(text: str) -> str:
        """Remove the CONFIDENCE line from the displayed advice."""
        return "\n".join(
            line for line in text.splitlines()
            if not line.strip().upper().startswith("CONFIDENCE:")
        ).strip()
