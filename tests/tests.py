import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
import pytest
from pawpal_system import Task, Pet, Owner, Scheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def owner():
    return Owner(name="Jordan", email="j@example.com", available_minutes_per_day=120)

@pytest.fixture
def pet():
    return Pet(name="Mochi", species="Dog", age_years=3)

@pytest.fixture
def scheduler(owner, pet):
    owner.add_pet(pet)
    return Scheduler(owner=owner, pet=pet)


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------

def test_task_is_completed_after_complete():
    task = Task(name="Walk", duration_minutes=30, priority=5, category="Exercise")
    assert task.is_completed is False
    task.complete()
    assert task.is_completed is True


def test_pet_task_count_increases_when_task_added():
    pet = Pet(name="Mochi", species="Dog", age_years=3)
    assert len(pet.get_tasks()) == 0
    pet.add_task(Task(name="Breakfast", duration_minutes=10, priority=5, category="Feeding"))
    assert len(pet.get_tasks()) == 1


# ---------------------------------------------------------------------------
# 1. Sorting correctness
# ---------------------------------------------------------------------------

def test_sort_by_time_returns_chronological_order(scheduler, pet):
    pet.add_task(Task("Evening Walk",  20, 3, "Exercise", scheduled_time="18:00"))
    pet.add_task(Task("Lunch",         10, 4, "Feeding",  scheduled_time="12:00"))
    pet.add_task(Task("Morning Walk",  30, 5, "Exercise", scheduled_time="07:00"))
    scheduler.generate_plan()

    sorted_tasks = scheduler.sort_by_time()
    times = [t.scheduled_time for t in sorted_tasks]
    assert times == ["07:00", "12:00", "18:00"]


def test_sort_by_time_tasks_without_time_go_last(scheduler, pet):
    pet.add_task(Task("Walk",      30, 5, "Exercise", scheduled_time="08:00"))
    pet.add_task(Task("Grooming",  15, 2, "Grooming", scheduled_time=""))
    pet.add_task(Task("Feeding",   10, 5, "Feeding",  scheduled_time="07:00"))
    scheduler.generate_plan()

    sorted_tasks = scheduler.sort_by_time()
    assert sorted_tasks[-1].name == "Grooming"
    assert sorted_tasks[0].scheduled_time == "07:00"


def test_sort_by_time_single_task(scheduler, pet):
    pet.add_task(Task("Walk", 30, 5, "Exercise", scheduled_time="09:00"))
    scheduler.generate_plan()
    assert len(scheduler.sort_by_time()) == 1


def test_sort_by_time_preserves_all_tasks(scheduler, pet):
    names = ["C-task", "A-task", "B-task"]
    times = ["14:00", "08:00", "11:00"]
    for name, time in zip(names, times):
        pet.add_task(Task(name, 10, 3, "Enrichment", scheduled_time=time))
    scheduler.generate_plan()

    sorted_tasks = scheduler.sort_by_time()
    assert len(sorted_tasks) == 3
    assert [t.name for t in sorted_tasks] == ["A-task", "B-task", "C-task"]


# ---------------------------------------------------------------------------
# 2. Recurrence logic
# ---------------------------------------------------------------------------

def test_daily_task_complete_sets_next_occurrence_tomorrow():
    task = Task("Walk", 30, 5, "Exercise", recurrence="Daily")
    before = datetime.now()
    task.complete()

    assert task.is_completed is True
    assert task.next_occurrence is not None
    expected_date = (before + timedelta(days=1)).date()
    assert task.next_occurrence.date() == expected_date


def test_weekly_task_complete_sets_next_occurrence_next_week():
    task = Task("Bath", 20, 3, "Grooming", recurrence="Weekly")
    before = datetime.now()
    task.complete()

    assert task.next_occurrence is not None
    expected_date = (before + timedelta(weeks=1)).date()
    assert task.next_occurrence.date() == expected_date


def test_non_recurring_task_complete_has_no_next_occurrence():
    task = Task("One-off Vet Visit", 60, 5, "Medical")
    task.complete()
    assert task.next_occurrence is None


def test_reset_clears_next_occurrence():
    task = Task("Walk", 30, 5, "Exercise", recurrence="Daily")
    task.complete()
    assert task.next_occurrence is not None
    task.reset()
    assert task.next_occurrence is None
    assert task.is_completed is False


def test_daily_task_next_occurrence_is_one_day_ahead():
    task = Task("Walk", 30, 5, "Exercise", recurrence="Daily")
    task.complete()
    delta = task.next_occurrence - datetime.now()
    assert timedelta(hours=23, minutes=59) < delta < timedelta(days=1, seconds=5)


# ---------------------------------------------------------------------------
# 3. Conflict detection
# ---------------------------------------------------------------------------

def test_detect_conflicts_returns_warning_for_duplicate_times(scheduler, pet):
    pet.add_task(Task("Walk",    30, 5, "Exercise", scheduled_time="08:00"))
    pet.add_task(Task("Feeding", 10, 5, "Feeding",  scheduled_time="08:00"))
    scheduler.generate_plan()

    warnings = scheduler.detect_conflicts()
    assert len(warnings) == 1
    assert "08:00" in warnings[0]
    assert "Walk" in warnings[0]
    assert "Feeding" in warnings[0]


def test_detect_conflicts_no_warning_when_times_are_unique(scheduler, pet):
    pet.add_task(Task("Walk",    30, 5, "Exercise", scheduled_time="07:00"))
    pet.add_task(Task("Feeding", 10, 5, "Feeding",  scheduled_time="08:00"))
    scheduler.generate_plan()

    assert scheduler.detect_conflicts() == []


def test_detect_conflicts_ignores_tasks_without_scheduled_time(scheduler, pet):
    pet.add_task(Task("Walk",    30, 5, "Exercise", scheduled_time=""))
    pet.add_task(Task("Feeding", 10, 5, "Feeding",  scheduled_time=""))
    scheduler.generate_plan()

    assert scheduler.detect_conflicts() == []


def test_detect_conflicts_does_not_raise(scheduler, pet):
    pet.add_task(Task("A", 10, 5, "Feeding",    scheduled_time="09:00"))
    pet.add_task(Task("B", 10, 4, "Exercise",   scheduled_time="09:00"))
    pet.add_task(Task("C", 10, 3, "Enrichment", scheduled_time="09:00"))
    scheduler.generate_plan()

    warnings = scheduler.detect_conflicts()
    assert isinstance(warnings, list)
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

def test_pet_with_no_tasks_returns_empty_list(pet):
    assert pet.get_tasks() == []


def test_scheduler_generate_plan_on_empty_pet(scheduler):
    plan = scheduler.generate_plan()
    assert plan == []
    assert scheduler.total_scheduled_minutes == 0


def test_sort_by_time_on_empty_plan(scheduler):
    scheduler.generate_plan()
    assert scheduler.sort_by_time() == []


def test_detect_conflicts_on_empty_plan(scheduler):
    scheduler.generate_plan()
    assert scheduler.detect_conflicts() == []


def test_filter_by_pet_name_unknown_pet_returns_empty(scheduler):
    assert scheduler.filter_by_pet_name("Ghost") == []


def test_filter_by_completion_all_incomplete(scheduler, pet):
    pet.add_task(Task("Walk",    30, 5, "Exercise"))
    pet.add_task(Task("Feeding", 10, 5, "Feeding"))
    assert len(scheduler.filter_by_completion(False)) == 2
    assert scheduler.filter_by_completion(True) == []


def test_remove_task_reduces_count(pet):
    pet.add_task(Task("Walk",    30, 5, "Exercise"))
    pet.add_task(Task("Feeding", 10, 5, "Feeding"))
    pet.remove_task("Walk")
    names = [t.name for t in pet.get_tasks()]
    assert "Walk" not in names
    assert len(names) == 1


def test_generate_plan_excludes_completed_tasks(scheduler, pet):
    done = Task("Old Walk", 30, 5, "Exercise")
    done.complete()
    pet.add_task(done)
    pet.add_task(Task("Feeding", 10, 5, "Feeding"))

    plan = scheduler.generate_plan()
    names = [t.name for t in plan]
    assert "Old Walk" not in names
    assert "Feeding" in names


def test_generate_plan_respects_time_budget():
    owner = Owner("Sam", "s@example.com", available_minutes_per_day=20)
    pet = Pet("Rex", "Dog", 2)
    owner.add_pet(pet)
    pet.add_task(Task("Long Walk",  60, 5, "Exercise"))
    pet.add_task(Task("Quick Feed", 10, 4, "Feeding"))

    sched = Scheduler(owner=owner, pet=pet)
    plan = sched.generate_plan()

    assert all(t.name != "Long Walk" for t in plan)
    assert sched.total_scheduled_minutes <= 20


def test_reset_plan_clears_state(scheduler, pet):
    pet.add_task(Task("Walk", 30, 5, "Exercise"))
    scheduler.generate_plan()
    assert len(scheduler.scheduled_tasks) > 0

    scheduler.reset_plan()
    assert scheduler.scheduled_tasks == []
    assert scheduler.total_scheduled_minutes == 0
