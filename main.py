from pawpal_system import Owner, Pet, Task, Scheduler

# --- Setup ---

owner = Owner(
    name="Jordan",
    email="jordan@example.com",
    available_minutes_per_day=90,
    preferences=["morning walks", "no late-night feeding"],
)

mochi = Pet(name="Mochi", species="Dog", age_years=3)
luna  = Pet(name="Luna",  species="Cat", age_years=5, health_notes="Needs joint supplement with food")

# --- Tasks for Mochi — added out of chronological order intentionally ---

mochi.add_task(Task(name="Obedience Training", duration_minutes=20, priority=3, category="Enrichment",  scheduled_time="14:00"))
mochi.add_task(Task(name="Morning Walk",       duration_minutes=30, priority=5, category="Exercise",    scheduled_time="07:00", recurrence="Daily"))
mochi.add_task(Task(name="Brush Coat",         duration_minutes=15, priority=2, category="Grooming",    scheduled_time="11:00"))
mochi.add_task(Task(name="Breakfast",          duration_minutes=10, priority=5, category="Feeding",     scheduled_time="07:00"))  # conflict with Morning Walk

# --- Tasks for Luna ---

luna.add_task(Task(name="Playtime",               duration_minutes=15, priority=4, category="Enrichment", scheduled_time="10:00"))
luna.add_task(Task(name="Breakfast + Supplement", duration_minutes=10, priority=5, category="Feeding",    scheduled_time="08:00", recurrence="Daily"))
luna.add_task(Task(name="Litter Box Clean",        duration_minutes=5,  priority=4, category="Hygiene",    scheduled_time="09:00"))
luna.add_task(Task(name="Nail Trim",               duration_minutes=10, priority=2, category="Grooming",   scheduled_time="15:00", recurrence="Weekly"))

owner.add_pet(mochi)
owner.add_pet(luna)

# --- Schedule ---

mochi_scheduler = Scheduler(owner=owner, pet=mochi)
luna_scheduler  = Scheduler(owner=owner, pet=luna)

mochi_scheduler.generate_plan()
luna_scheduler.generate_plan()

# --- Formatting helpers ---

DIVIDER      = "=" * 56
THIN_DIVIDER = "-" * 56

def print_section(title: str) -> None:
    print()
    print(f"  {title}")
    print(THIN_DIVIDER)

def format_task_row(i: int, task) -> str:
    flag      = " !" if task.is_high_priority() else "  "
    time_part = f" @ {task.scheduled_time}" if task.scheduled_time else "        "
    recur     = f" [{task.recurrence}]" if task.recurrence else ""
    return (
        f"{flag} {i}. {task.name:<26}{time_part}  "
        f"{task.duration_minutes:>3} min  [P{task.priority}]{recur}"
    )

# ---------------------------------------------------------------------------
# Section 1 — Conflict warnings
# ---------------------------------------------------------------------------

print()
print(DIVIDER)
print("           PAWPAL+  —  ALGORITHM REPORT")
print(DIVIDER)

print_section("CONFLICT DETECTION")
for scheduler, label in ((mochi_scheduler, "Mochi"), (luna_scheduler, "Luna")):
    conflicts = scheduler.detect_conflicts()
    if conflicts:
        for warning in conflicts:
            print(f"  !  {label}: {warning}")
    else:
        print(f"  ok {label}: no conflicts")

# ---------------------------------------------------------------------------
# Section 2 — Time-sorted schedule
# ---------------------------------------------------------------------------

print_section("SORTED SCHEDULE  (chronological order)")

for scheduler in (mochi_scheduler, luna_scheduler):
    pet = scheduler.pet
    sorted_tasks = scheduler.sort_by_time()
    print(f"\n  {pet.name}  ({pet.species})")
    if sorted_tasks:
        for i, task in enumerate(sorted_tasks, start=1):
            print(format_task_row(i, task))
    else:
        print("    No tasks scheduled.")
    print(f"\n  Time used: {scheduler.total_scheduled_minutes} / "
          f"{scheduler.owner.available_minutes_per_day} min")

# ---------------------------------------------------------------------------
# Section 3 — Filtered task list for one pet
# ---------------------------------------------------------------------------

FILTER_PET = "Mochi"
print_section(f"FILTERED TASKS  —  {FILTER_PET} only")

filtered = mochi_scheduler.filter_by_pet_name(FILTER_PET)
if filtered:
    for i, task in enumerate(filtered, start=1):
        print(format_task_row(i, task))
else:
    print(f"  No tasks found for {FILTER_PET}.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

print()
print(f"  Owner: {owner.name}   |   Daily budget per pet: {owner.available_minutes_per_day} min")
print(f"  ! = high priority   [Daily]/[Weekly] = recurring task")
print(DIVIDER)
print()
