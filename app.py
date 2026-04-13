import streamlit as st
import pandas as pd
from pawpal_system import Owner, Pet, Task, Scheduler

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

if "owner" not in st.session_state:
    st.session_state.owner = Owner(
        name="Jordan",
        email="owner@example.com",
        available_minutes_per_day=90,
    )
if "show_schedule" not in st.session_state:
    st.session_state.show_schedule = False

owner: Owner = st.session_state.owner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_MAP   = {"Low (1)": 1, "Medium (3)": 3, "High (5)": 5}
PRIORITY_STARS = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
CATEGORIES     = ["Exercise", "Feeding", "Grooming", "Enrichment", "Hygiene", "Medication", "Other"]
RECURRENCES    = ["None", "Daily", "Weekly"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tasks_to_df(tasks: list) -> pd.DataFrame:
    """Convert a list of Task objects to a display DataFrame."""
    if not tasks:
        return pd.DataFrame()
    rows = []
    for t in tasks:
        rows.append({
            "Task":        t.name,
            "Category":    t.category,
            "Start":       t.scheduled_time or "—",
            "Duration":    f"{t.duration_minutes} min",
            "Priority":    PRIORITY_STARS.get(t.priority, str(t.priority)),
            "Recurrence":  t.recurrence or "—",
            "Status":      "Done" if t.is_completed else "Pending",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("PawPal+")
st.caption("Smart daily care planning for your pets.")
st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Owner profile
# ---------------------------------------------------------------------------

st.subheader("Owner Profile")

col1, col2 = st.columns(2)
with col1:
    new_name = st.text_input("Your name", value=owner.name)
    if new_name != owner.name:
        owner.name = new_name

with col2:
    new_budget = st.number_input(
        "Daily time budget (min)",
        min_value=10, max_value=480,
        value=owner.available_minutes_per_day,
        step=5,
    )
    if new_budget != owner.available_minutes_per_day:
        owner.set_availability(int(new_budget))

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Add a pet
# ---------------------------------------------------------------------------

st.subheader("Add a Pet")

with st.form("add_pet_form", clear_on_submit=True):
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        pet_name = st.text_input("Pet name", placeholder="e.g. Mochi")
    with pcol2:
        species = st.selectbox("Species", ["Dog", "Cat", "Rabbit", "Bird", "Other"])
    with pcol3:
        age = st.number_input("Age (years)", min_value=0, max_value=30, value=1)
    health_notes = st.text_input("Health notes (optional)", placeholder="e.g. needs joint supplement")
    submitted_pet = st.form_submit_button("Add Pet")

if submitted_pet:
    if not pet_name.strip():
        st.warning("Please enter a pet name.")
    elif pet_name.strip() in [p.name for p in owner.get_pets()]:
        st.warning(f"A pet named '{pet_name.strip()}' already exists.")
    else:
        owner.add_pet(Pet(
            name=pet_name.strip(),
            species=species,
            age_years=int(age),
            health_notes=health_notes.strip(),
        ))
        st.success(f"Added {pet_name.strip()} the {species}!")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Add a task
# ---------------------------------------------------------------------------

st.subheader("Add a Task")

pets = owner.get_pets()

if not pets:
    st.info("Add a pet above before adding tasks.")
else:
    with st.form("add_task_form", clear_on_submit=True):
        target_pet_name = st.selectbox("Assign to pet", [p.name for p in pets])

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            task_name = st.text_input("Task name", placeholder="e.g. Morning Walk")
        with row1_col2:
            category = st.selectbox("Category", CATEGORIES)
        with row1_col3:
            priority_label = st.selectbox("Priority", list(PRIORITY_MAP.keys()), index=1)

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            duration = st.number_input("Duration (min)", min_value=1, max_value=240, value=20)
        with row2_col2:
            scheduled_time = st.text_input("Start time (HH:MM)", placeholder="e.g. 08:30")
        with row2_col3:
            recurrence_label = st.selectbox("Recurrence", RECURRENCES)

        task_notes = st.text_input("Notes (optional)")
        submitted_task = st.form_submit_button("Add Task")

    if submitted_task:
        if not task_name.strip():
            st.warning("Please enter a task name.")
        else:
            target_pet = next(p for p in pets if p.name == target_pet_name)
            recurrence = "" if recurrence_label == "None" else recurrence_label
            target_pet.add_task(Task(
                name=task_name.strip(),
                duration_minutes=int(duration),
                priority=PRIORITY_MAP[priority_label],
                category=category,
                notes=task_notes.strip(),
                scheduled_time=scheduled_time.strip(),
                recurrence=recurrence,
            ))
            st.success(f"Added '{task_name.strip()}' to {target_pet_name}'s tasks.")
            st.session_state.show_schedule = False

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Pets, tasks, and task actions
# ---------------------------------------------------------------------------

st.subheader("Your Pets & Tasks")

pets = owner.get_pets()

if not pets:
    st.info("No pets added yet.")
else:
    for pet in pets:
        tasks = pet.get_tasks()
        task_count = len(tasks)
        done_count = sum(1 for t in tasks if t.is_completed)

        header = f"**{pet.name}** — {pet.species}, age {pet.age_years}  ·  {task_count} task(s), {done_count} done"
        with st.expander(header, expanded=True):
            if pet.health_notes:
                st.caption(f"Health note: {pet.health_notes}")

            if not tasks:
                st.write("No tasks yet.")
            else:
                st.dataframe(tasks_to_df(tasks), use_container_width=True, hide_index=True)

                pending  = [t for t in tasks if not t.is_completed]
                done     = [t for t in tasks if t.is_completed]

                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    if pending:
                        task_to_complete = st.selectbox(
                            "Mark complete",
                            [t.name for t in pending],
                            key=f"sel_complete_{pet.name}",
                        )
                        if st.button("Mark Done", key=f"btn_complete_{pet.name}"):
                            task_obj = next(t for t in pet.tasks if t.name == task_to_complete)
                            task_obj.complete()
                            next_occ = task_obj.next_occurrence
                            if next_occ:
                                st.success(
                                    f"'{task_to_complete}' done! "
                                    f"Next occurrence: {next_occ.strftime('%a %b %d, %Y')}"
                                )
                            else:
                                st.success(f"'{task_to_complete}' marked as done.")
                            st.session_state.show_schedule = False
                            st.rerun()

                with action_col2:
                    if done:
                        task_to_reset = st.selectbox(
                            "Reset task",
                            [t.name for t in done],
                            key=f"sel_reset_{pet.name}",
                        )
                        if st.button("Reset", key=f"btn_reset_{pet.name}"):
                            task_obj = next(t for t in pet.tasks if t.name == task_to_reset)
                            task_obj.reset()
                            st.info(f"'{task_to_reset}' reset to pending.")
                            st.session_state.show_schedule = False
                            st.rerun()

            st.divider()
            remove_col, _ = st.columns([1, 4])
            with remove_col:
                if st.button(f"Remove {pet.name}", key=f"remove_{pet.name}"):
                    owner.remove_pet(pet.name)
                    st.session_state.show_schedule = False
                    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Live conflict check — runs on every render, no button needed
# ---------------------------------------------------------------------------

pets = owner.get_pets()
all_conflicts: list[tuple[str, str]] = []

for pet in pets:
    if len([t for t in pet.get_tasks() if t.scheduled_time]) >= 2:
        sched = Scheduler(owner=owner, pet=pet)
        sched.generate_plan()
        for warning in sched.detect_conflicts():
            all_conflicts.append((pet.name, warning))

if all_conflicts:
    st.subheader("Schedule Conflicts Detected")
    for pet_name, warning in all_conflicts:
        st.warning(f"**{pet_name}:** {warning}")
    st.caption("Two or more tasks share the same start time. Edit a task's start time to resolve.")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Generate schedule
# ---------------------------------------------------------------------------

st.subheader("Today's Schedule")

pets = owner.get_pets()
has_tasks = any(pet.get_tasks() for pet in pets)

if not pets:
    st.info("Add at least one pet to generate a schedule.")
elif not has_tasks:
    st.info("Add at least one task to generate a schedule.")
else:
    if st.button("Generate Schedule", type="primary"):
        st.session_state.show_schedule = True

    if st.session_state.show_schedule:
        st.markdown(f"**Owner:** {owner.name} · **Daily budget per pet:** {owner.available_minutes_per_day} min")
        st.write("")

        for pet in pets:
            if not pet.get_tasks():
                continue

            scheduler = Scheduler(owner=owner, pet=pet)
            plan      = scheduler.generate_plan()
            sorted_plan = scheduler.sort_by_time()
            conflicts = scheduler.detect_conflicts()
            deferred  = scheduler.get_unscheduled_tasks()
            used      = scheduler.total_scheduled_minutes
            budget    = owner.available_minutes_per_day

            st.markdown(f"#### {pet.name}  ({pet.species})")

            for warning in conflicts:
                st.warning(f"{warning}")

            progress_pct = min(used / budget, 1.0) if budget > 0 else 0
            budget_label = f"Time used: {used} / {budget} min  ({int(progress_pct * 100)}%)"
            st.progress(progress_pct, text=budget_label)

            if sorted_plan:
                st.dataframe(tasks_to_df(sorted_plan), use_container_width=True, hide_index=True)
            else:
                st.warning("No incomplete tasks fit within the daily budget.")

            if deferred:
                with st.expander(f"Deferred tasks ({len(deferred)})", expanded=False):
                    st.dataframe(tasks_to_df(deferred), use_container_width=True, hide_index=True)

            completed = scheduler.filter_by_completion(completed=True)
            if completed:
                with st.expander(f"Completed tasks ({len(completed)})", expanded=False):
                    st.dataframe(tasks_to_df(completed), use_container_width=True, hide_index=True)

            st.write("")

        st.divider()

        st.markdown("#### Filter Tasks by Pet")
        all_pet_names = [p.name for p in owner.get_pets() if p.get_tasks()]

        if all_pet_names:
            ref_scheduler = Scheduler(owner=owner, pet=owner.get_pets()[0])
            filter_pet = st.selectbox("Select a pet to inspect", all_pet_names, key="filter_pet_select")
            filtered   = ref_scheduler.filter_by_pet_name(filter_pet)

            if filtered:
                pending_filtered   = [t for t in filtered if not t.is_completed]
                completed_filtered = [t for t in filtered if t.is_completed]

                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.metric("Total tasks", len(filtered))
                with fcol2:
                    st.metric("Completed", f"{len(completed_filtered)} / {len(filtered)}")

                st.dataframe(tasks_to_df(filtered), use_container_width=True, hide_index=True)
            else:
                st.info(f"No tasks found for {filter_pet}.")

st.divider()

# ---------------------------------------------------------------------------
# Section 6 — AI Care Advisor (Agentic RAG)
# ---------------------------------------------------------------------------

st.subheader("AI Care Advisor")
st.caption(
    "Claude analyses your pet's schedule, retrieves evidence-based care guidelines, "
    "identifies any gaps, and gives personalised recommendations."
)

pets_with_tasks = [p for p in owner.get_pets() if p.get_tasks()]

if not pets_with_tasks:
    st.info("Add a pet and at least one task to unlock AI care advice.")
else:
    # Initialise the advisor once per session
    if "ai_advisor" not in st.session_state:
        try:
            from ai_advisor import AIAdvisor
            st.session_state.ai_advisor = AIAdvisor()
            st.session_state.ai_ready = True
        except Exception as exc:
            st.session_state.ai_ready = False
            st.session_state.ai_init_error = str(exc)

    if st.session_state.get("ai_ready", False):
        adv_col1, adv_col2 = st.columns([3, 1])
        with adv_col1:
            advisor_pet_name = st.selectbox(
                "Select pet for advice",
                [p.name for p in pets_with_tasks],
                key="advisor_pet_select",
            )
        with adv_col2:
            st.write("")  # vertical spacer
            get_advice = st.button("Get Advice", type="secondary", key="get_advice_btn")

        if get_advice:
            target_pet = next(p for p in pets_with_tasks if p.name == advisor_pet_name)
            sched = Scheduler(owner=owner, pet=target_pet)
            plan = sched.generate_plan()

            with st.spinner(f"Analysing {target_pet.name}'s schedule..."):
                result = st.session_state.ai_advisor.get_advice(target_pet, plan, owner)

            st.session_state.last_advice = result
            st.session_state.last_advice_pet = advisor_pet_name

        if "last_advice" in st.session_state:
            result = st.session_state.last_advice
            advised_pet = st.session_state.get("last_advice_pet", "")

            st.markdown(f"**Advice for {advised_pet}**")

            if result.get("error"):
                st.error(
                    f"AI advice unavailable: {result['advice']}\n\n"
                    "Make sure the `ANTHROPIC_API_KEY` environment variable is set."
                )
            else:
                conf = result["confidence"]
                conf_label = "High" if conf >= 0.8 else "Medium" if conf >= 0.5 else "Low"

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Confidence", f"{int(conf * 100)}%")
                with mc2:
                    st.metric("Reliability", conf_label)
                with mc3:
                    st.metric("Tools called", len(result["tools_used"]))

                if result["gaps_found"]:
                    st.warning(
                        "Care gaps were identified in today's schedule — "
                        "see the recommendations below."
                    )

                st.markdown(result["advice"])

                with st.expander("Advisor diagnostics", expanded=False):
                    st.write("**Tools Claude called:**", result["tools_used"])
                    st.write(f"**Response time:** {result['elapsed_seconds']:.1f} s")
                    st.caption("Full interaction log: logs/pawpal_ai.log")
    else:
        st.warning(
            "AI advisor could not be initialised. "
            "Ensure the `ANTHROPIC_API_KEY` environment variable is set and the "
            "`anthropic` package is installed.\n\n"
            f"Error: {st.session_state.get('ai_init_error', 'unknown')}"
        )
