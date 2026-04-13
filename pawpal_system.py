from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import ClassVar, List


@dataclass
class Task:
    """A single pet care activity with a duration and priority level.

    Priority scale: 1 (lowest) to 5 (highest).
    Tasks with priority >= HIGH_PRIORITY_THRESHOLD are considered high priority.

    Optional fields:
      scheduled_time  — planned start time as 'HH:MM' string (empty = unscheduled)
      recurrence      — '' | 'Daily' | 'Weekly'; triggers next_occurrence on complete()
      next_occurrence — set automatically by complete() for recurring tasks
    """

    HIGH_PRIORITY_THRESHOLD: ClassVar[int] = 3

    name: str
    duration_minutes: int
    priority: int
    category: str
    is_completed: bool = False
    notes: str = ""
    scheduled_time: str = ""          # 'HH:MM' or empty
    recurrence: str = ""              # '' | 'Daily' | 'Weekly'
    next_occurrence: datetime | None = field(default=None, compare=False, repr=False)

    def complete(self) -> None:
        """Mark task complete and compute next_occurrence for Daily/Weekly recurrences."""
        self.is_completed = True
        if self.recurrence == "Daily":
            self.next_occurrence = datetime.now() + timedelta(days=1)
        elif self.recurrence == "Weekly":
            self.next_occurrence = datetime.now() + timedelta(weeks=1)
        else:
            self.next_occurrence = None

    def reset(self) -> None:
        """Set is_completed to False and clear next_occurrence."""
        self.is_completed = False
        self.next_occurrence = None

    def is_high_priority(self) -> bool:
        """Return True if priority meets or exceeds HIGH_PRIORITY_THRESHOLD."""
        return self.priority >= Task.HIGH_PRIORITY_THRESHOLD

    def summary(self) -> str:
        """Return a formatted one-line description of the task."""
        status = "done" if self.is_completed else "pending"
        time_part = f" @ {self.scheduled_time}" if self.scheduled_time else ""
        recur_part = f" [{self.recurrence}]" if self.recurrence else ""
        return (
            f"[{self.category}] {self.name}{time_part}{recur_part} — "
            f"{self.duration_minutes} min, priority {self.priority}/5 ({status})"
        )


@dataclass
class Pet:
    name: str
    species: str
    age_years: int
    health_notes: str = ""
    tasks: List[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Append a Task to this pet's task list."""
        self.tasks.append(task)

    def remove_task(self, task_name: str) -> None:
        """Remove all tasks whose name matches task_name."""
        self.tasks = [t for t in self.tasks if t.name != task_name]

    def get_tasks(self) -> List[Task]:
        """Return a shallow copy of all tasks for this pet."""
        return list(self.tasks)

    def get_high_priority_tasks(self) -> List[Task]:
        """Return tasks where is_high_priority() is True."""
        return [t for t in self.tasks if t.is_high_priority()]


class Owner:
    def __init__(
        self,
        name: str,
        email: str,
        available_minutes_per_day: int,
        preferences: List[str] | None = None,
    ) -> None:
        self.name = name
        self.email = email
        self.available_minutes_per_day = available_minutes_per_day
        self.preferences: List[str] = preferences if preferences is not None else []
        self._pets: List[Pet] = []

    def add_pet(self, pet: Pet) -> None:
        """Append a Pet to this owner's pet list."""
        self._pets.append(pet)

    def remove_pet(self, pet_name: str) -> None:
        """Remove all pets whose name matches pet_name."""
        self._pets = [p for p in self._pets if p.name != pet_name]

    def get_pets(self) -> List[Pet]:
        """Return a shallow copy of all pets belonging to this owner."""
        return list(self._pets)

    def set_availability(self, minutes: int) -> None:
        """Set available_minutes_per_day to the given value."""
        self.available_minutes_per_day = minutes


class Scheduler:
    def __init__(self, owner: Owner, pet: Pet) -> None:
        self.owner = owner
        self.pet = pet
        self.scheduled_tasks: List[Task] = []
        self.total_scheduled_minutes: int = 0

    def reset_plan(self) -> None:
        """Clear scheduled_tasks and reset total_scheduled_minutes to zero."""
        self.scheduled_tasks = []
        self.total_scheduled_minutes = 0

    def get_all_owner_tasks(self) -> List[Task]:
        """Collect and return all tasks across every pet the owner has, sorted by priority descending."""
        all_tasks: List[Task] = []
        for pet in self.owner.get_pets():
            all_tasks.extend(pet.get_tasks())
        return sorted(all_tasks, key=lambda t: t.priority, reverse=True)

    def generate_plan(self) -> List[Task]:
        """Greedily select incomplete tasks by priority until the owner's time budget is filled."""
        self.reset_plan()
        candidates = sorted(
            [t for t in self.pet.get_tasks() if not t.is_completed],
            key=lambda t: t.priority,
            reverse=True,
        )
        for task in candidates:
            if self.total_scheduled_minutes + task.duration_minutes <= self.owner.available_minutes_per_day:
                self.scheduled_tasks.append(task)
                self.total_scheduled_minutes += task.duration_minutes
        return list(self.scheduled_tasks)

    def sort_by_time(self) -> List[Task]:
        """Return scheduled_tasks sorted ascending by scheduled_time ('HH:MM'); unscheduled tasks go last."""
        def time_key(task: Task) -> datetime:
            if task.scheduled_time:
                return datetime.strptime(task.scheduled_time, "%H:%M")
            return datetime.max

        return sorted(self.scheduled_tasks, key=lambda t: time_key(t))

    def filter_by_pet_name(self, pet_name: str) -> List[Task]:
        """Return all tasks belonging to the owner's pet whose name matches pet_name."""
        for pet in self.owner.get_pets():
            if pet.name == pet_name:
                return pet.get_tasks()
        return []

    def filter_by_completion(self, completed: bool) -> List[Task]:
        """Return tasks from the current pet whose is_completed matches the given value."""
        return [t for t in self.pet.get_tasks() if t.is_completed == completed]

    def detect_conflicts(self) -> List[str]:
        """Return a list of warning strings for any two scheduled tasks sharing the same start time."""
        warnings: List[str] = []
        seen: dict[str, str] = {}
        for task in self.scheduled_tasks:
            if not task.scheduled_time:
                continue
            if task.scheduled_time in seen:
                warnings.append(
                    f"Conflict at {task.scheduled_time}: "
                    f"'{seen[task.scheduled_time]}' and '{task.name}' overlap."
                )
            else:
                seen[task.scheduled_time] = task.name
        return warnings

    def filter_by_priority(self, min_priority: int) -> List[Task]:
        """Return tasks from the current pet whose priority is at or above min_priority."""
        return [t for t in self.pet.get_tasks() if t.priority >= min_priority]

    def fits_in_available_time(self, tasks: List[Task]) -> bool:
        """Return True if the total duration of the given tasks fits within the owner's daily budget."""
        return sum(t.duration_minutes for t in tasks) <= self.owner.available_minutes_per_day

    def explain_plan(self) -> str:
        """Return a formatted string summarising scheduled tasks and any deferred tasks."""
        if not self.scheduled_tasks:
            return "No plan generated yet. Call generate_plan() first."

        lines = [
            f"Plan for {self.pet.name} — {self.total_scheduled_minutes} of "
            f"{self.owner.available_minutes_per_day} available minutes used.\n"
        ]
        for i, task in enumerate(self.scheduled_tasks, start=1):
            time_part = f" @ {task.scheduled_time}" if task.scheduled_time else ""
            lines.append(
                f"  {i}. {task.name}{time_part} ({task.duration_minutes} min, priority {task.priority}/5)"
            )

        unscheduled = self.get_unscheduled_tasks()
        if unscheduled:
            lines.append("\nExcluded (insufficient time remaining):")
            for task in unscheduled:
                lines.append(f"  - {task.name} ({task.duration_minutes} min, priority {task.priority}/5)")

        conflicts = self.detect_conflicts()
        if conflicts:
            lines.append("\nConflicts detected:")
            for warning in conflicts:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)

    def get_unscheduled_tasks(self) -> List[Task]:
        """Return incomplete tasks from the current pet that were not selected by generate_plan()."""
        scheduled_names = {t.name for t in self.scheduled_tasks}
        return [
            t for t in self.pet.get_tasks()
            if not t.is_completed and t.name not in scheduled_names
        ]
