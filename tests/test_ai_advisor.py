"""
test_ai_advisor.py — Unit tests for the ai_advisor module.

Covers:
  - PetCareRAG.retrieve()          (knowledge base retrieval)
  - PetCareRAG.identify_gaps()     (gap detection)
  - AIAdvisor._parse_confidence()  (score parsing)
  - AIAdvisor._strip_confidence_line()
  - AIAdvisor._execute_tool()      (tool dispatch without API calls)
  - AIAdvisor.get_advice()         (full loop, Anthropic client mocked)

The Anthropic client is always mocked so no real API calls are made.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import MagicMock, patch

from pawpal_system import Owner, Pet, Task
from ai_advisor import (
    AIAdvisor,
    PetCareRAG,
    CARE_KNOWLEDGE_BASE,
    ESSENTIAL_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_advisor() -> AIAdvisor:
    """Return an AIAdvisor whose Anthropic client is replaced with a MagicMock."""
    with patch("ai_advisor.anthropic.Anthropic"):
        advisor = AIAdvisor()
    advisor.client = MagicMock()
    return advisor


def _end_turn_response(text: str) -> MagicMock:
    """Build a mock Anthropic response that ends the agentic loop immediately."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# 1. PetCareRAG — retrieval
# ---------------------------------------------------------------------------

class TestPetCareRAGRetrieve:
    def test_returns_list_of_strings_for_known_species_and_category(self):
        rag = PetCareRAG()
        tips = rag.retrieve("Dog", "Exercise")
        assert isinstance(tips, list)
        assert len(tips) > 0
        assert all(isinstance(t, str) for t in tips)

    def test_unknown_species_falls_back_to_other_entry(self):
        rag = PetCareRAG()
        tips = rag.retrieve("Dragon", "Feeding")
        assert isinstance(tips, list)
        assert len(tips) > 0

    def test_unknown_category_returns_single_fallback_string(self):
        rag = PetCareRAG()
        tips = rag.retrieve("Dog", "SkyDiving")
        assert len(tips) == 1
        assert "consult" in tips[0].lower() or "No specific" in tips[0]

    def test_all_known_species_have_feeding_tips(self):
        rag = PetCareRAG()
        for species in CARE_KNOWLEDGE_BASE:
            tips = rag.retrieve(species, "Feeding")
            assert len(tips) > 0, f"No feeding tips found for {species}"

    def test_cat_exercise_tips_mention_play(self):
        rag = PetCareRAG()
        tips = rag.retrieve("Cat", "Exercise")
        combined = " ".join(tips).lower()
        assert "play" in combined


# ---------------------------------------------------------------------------
# 2. PetCareRAG — gap detection
# ---------------------------------------------------------------------------

class TestPetCareRAGGaps:
    def test_detects_missing_essential_categories_for_dog(self):
        rag = PetCareRAG()
        # Dog essentials: Exercise, Feeding, Grooming
        gaps = rag.identify_gaps("Dog", ["Grooming"])
        gap_cats = [g["category"] for g in gaps]
        assert "Exercise" in gap_cats
        assert "Feeding" in gap_cats
        assert "Grooming" not in gap_cats

    def test_no_gaps_when_all_essentials_are_covered(self):
        rag = PetCareRAG()
        for species, essentials in ESSENTIAL_CATEGORIES.items():
            gaps = rag.identify_gaps(species, essentials)
            assert gaps == [], f"Unexpected gaps for {species}: {gaps}"

    def test_returns_list_of_dicts_with_required_keys(self):
        rag = PetCareRAG()
        gaps = rag.identify_gaps("Cat", [])
        for gap in gaps:
            assert "category" in gap
            assert "reason" in gap

    def test_unknown_species_uses_other_essentials(self):
        rag = PetCareRAG()
        gaps = rag.identify_gaps("Unicorn", [])
        # "Other" essentials = ["Feeding"]
        assert any(g["category"] == "Feeding" for g in gaps)

    def test_empty_schedule_produces_all_essential_gaps_for_rabbit(self):
        rag = PetCareRAG()
        gaps = rag.identify_gaps("Rabbit", [])
        gap_cats = {g["category"] for g in gaps}
        for essential in ESSENTIAL_CATEGORIES["Rabbit"]:
            assert essential in gap_cats


# ---------------------------------------------------------------------------
# 3. AIAdvisor — confidence parsing and text stripping
# ---------------------------------------------------------------------------

class TestConfidenceParsing:
    def test_extracts_score_from_last_line(self):
        advisor = _make_advisor()
        text = "Here is my advice.\n\nCONFIDENCE: 0.87"
        assert advisor._parse_confidence(text) == pytest.approx(0.87)

    def test_case_insensitive(self):
        advisor = _make_advisor()
        assert advisor._parse_confidence("confidence: 0.92") == pytest.approx(0.92)

    def test_clamps_score_above_one(self):
        advisor = _make_advisor()
        assert advisor._parse_confidence("CONFIDENCE: 1.50") == pytest.approx(1.0)

    def test_clamps_score_below_zero(self):
        advisor = _make_advisor()
        assert advisor._parse_confidence("CONFIDENCE: -0.3") == pytest.approx(0.0)

    def test_returns_default_when_line_is_absent(self):
        advisor = _make_advisor()
        assert advisor._parse_confidence("No score here.") == pytest.approx(0.75)

    def test_strip_removes_confidence_line(self):
        advisor = _make_advisor()
        text = "Recommendation 1.\nRecommendation 2.\nCONFIDENCE: 0.80"
        result = advisor._strip_confidence_line(text)
        assert "CONFIDENCE" not in result
        assert "Recommendation 1." in result


# ---------------------------------------------------------------------------
# 4. AIAdvisor — tool execution (no API calls)
# ---------------------------------------------------------------------------

class TestToolExecution:
    def test_retrieve_care_tips_returns_valid_json(self):
        advisor = _make_advisor()
        raw = advisor._execute_tool("retrieve_care_tips", {"species": "Dog", "category": "Exercise"})
        data = json.loads(raw)
        assert "tips" in data
        assert data["count"] > 0

    def test_identify_schedule_gaps_returns_valid_json(self):
        advisor = _make_advisor()
        raw = advisor._execute_tool(
            "identify_schedule_gaps",
            {"species": "Dog", "scheduled_categories": ["Grooming"]},
        )
        data = json.loads(raw)
        assert "gaps" in data
        assert "gap_count" in data

    def test_unknown_tool_returns_error_json(self):
        advisor = _make_advisor()
        raw = advisor._execute_tool("teleport", {})
        data = json.loads(raw)
        assert "error" in data


# ---------------------------------------------------------------------------
# 5. AIAdvisor.get_advice() — full loop with mocked Claude client
# ---------------------------------------------------------------------------

class TestGetAdvice:
    def _make_pet_owner(self):
        owner = Owner("Jordan", "j@example.com", 90)
        pet = Pet("Mochi", "Dog", 3)
        owner.add_pet(pet)
        task = Task("Morning Walk", 30, 5, "Exercise", scheduled_time="07:00")
        pet.add_task(task)
        return pet, owner, [task]

    def test_returns_dict_with_all_required_keys(self):
        advisor = _make_advisor()
        pet, owner, tasks = self._make_pet_owner()
        advisor.client.messages.create.return_value = _end_turn_response(
            "Great plan!\n\nCONFIDENCE: 0.85"
        )
        result = advisor.get_advice(pet, tasks, owner)
        for key in ("advice", "confidence", "tools_used", "gaps_found", "elapsed_seconds"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_parsed_correctly(self):
        advisor = _make_advisor()
        pet, owner, tasks = self._make_pet_owner()
        advisor.client.messages.create.return_value = _end_turn_response(
            "Some advice here.\n\nCONFIDENCE: 0.91"
        )
        result = advisor.get_advice(pet, tasks, owner)
        assert result["confidence"] == pytest.approx(0.91)

    def test_confidence_line_not_in_advice_text(self):
        advisor = _make_advisor()
        pet, owner, tasks = self._make_pet_owner()
        advisor.client.messages.create.return_value = _end_turn_response(
            "Feed twice daily.\n\nCONFIDENCE: 0.80"
        )
        result = advisor.get_advice(pet, tasks, owner)
        assert "CONFIDENCE" not in result["advice"]

    def test_api_error_returns_error_dict(self):
        import anthropic as _anthropic
        advisor = _make_advisor()
        pet, owner, tasks = self._make_pet_owner()
        advisor.client.messages.create.side_effect = _anthropic.APIStatusError(
            "rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        result = advisor.get_advice(pet, tasks, owner)
        assert "error" in result
        assert result["confidence"] == pytest.approx(0.0)
