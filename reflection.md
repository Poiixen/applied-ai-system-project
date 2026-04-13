# PawPal+ Project Reflection

## 1. System Design
- Register owner and pet profiles; things like basic information about themselves and their pets. 
- Add and manage care tasts; user creates tasks that includes duration and priority level, they can edit or remove them. The task list is the primary input
- Generate view w/ daily plan; schedule generation, and the app produces an ordered daily plan that fits within the owner's available time, respects task priorities, and explains the reasoning behind the chosen order.

**a. Initial design**

Four classes: `Task` (care activity with duration and priority), `Pet` (profile + task list), `Owner` (profile + pet list + daily availability), and `Scheduler` (selects and orders tasks within the owner's time budget). `Task` and `Pet` use `@dataclass`; `Owner` and `Scheduler` use plain classes because they hold private derived state.

**b. Design changes**

Yes; two fixes made during skeleton review:

1. Added `HIGH_PRIORITY_THRESHOLD: ClassVar[int] = 3` to `Task`. The original design defined `is_high_priority()` but had no numeric cutoff, which would cause inconsistent implementations across all classes that rely on priority comparisons.
2. Added `reset_plan()` to `Scheduler`. Without it, calling `generate_plan()` more than once would accumulate stale tasks in `scheduled_tasks` — a clear state management bug before any logic is even written.

---

## 2. Scheduling Logic and Tradeoffs

**a. Constraints and priorities**

- What constraints does your scheduler consider (for example: time, priority, preferences)?
- How did you decide which constraints mattered most?

**b. Tradeoffs**

Conflict detection checks for exact `scheduled_time` string matches only — it does not compute whether two tasks' time windows overlap based on duration. A 30-minute walk at `07:00` and a 10-minute feeding at `07:00` trigger a warning, but a walk at `07:00` and a feeding at `07:15` do not, even though they overlap. This is a reasonable tradeoff for a lightweight pet care planner: most tasks are loosely time-boxed, the owner is the one actually executing them, and full interval-overlap checking would add complexity (sorting by start time, comparing start + duration against the next task's start) without meaningful benefit at this scale.

---

## 3. AI Collaboration

**a. How you used AI**

- How did you use AI tools during this project (for example: design brainstorming, debugging, refactoring)?
- What kinds of prompts or questions were most helpful?

**b. Judgment and verification**

- Describe one moment where you did not accept an AI suggestion as-is.
- How did you evaluate or verify what the AI suggested?

---

## 4. Testing and Verification

**a. What you tested**

- What behaviors did you test?
- Why were these tests important?

**b. Confidence**

- How confident are you that your scheduler works correctly?
- What edge cases would you test next if you had more time?

---

## 5. Reflection

**a. What went well**

- What part of this project are you most satisfied with?

**b. What you would improve**

- If you had another iteration, what would you improve or redesign?

**c. Key takeaway**

- What is one important thing you learned about designing systems or working with AI on this project?
