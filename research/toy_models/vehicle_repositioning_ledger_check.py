"""Bounded deterministic bookkeeping checks; no optimizer or routing validation.

Run with Python -B. All times are absolute, arrivals are usable at their end
time, and each immutable task appears exactly once in the ledger.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    kind: str
    origin: int
    destination: int
    start: int
    end: int
    vehicles: int = 1
    cargo: float = 0.0


def valid_task(task, tau=2):
    if task.kind not in {"local", "loaded", "empty"}:
        return False
    if task.origin not in (1, 2) or task.destination not in (1, 2):
        return False
    if task.end <= task.start or task.vehicles <= 0 or task.cargo < 0:
        return False
    if task.kind == "local":
        return task.origin == task.destination
    if task.origin == task.destination:
        return False
    if task.kind == "empty":
        return task.end - task.start == tau and task.cargo == 0
    # A capacity upper bound alone does NOT imply positive cargo per vehicle.
    return True


def state(initial, tasks, time):
    """Return idle-by-city, active local count, intercity in-transit count."""
    idle = dict(initial)
    local = transit = 0
    for task in tasks:
        if task.kind == "local":
            if task.start <= time < task.end:
                idle[task.origin] -= task.vehicles
                local += task.vehicles
        else:
            if task.start <= time:
                idle[task.origin] -= task.vehicles
            if task.end <= time:
                idle[task.destination] += task.vehicles
            if task.start <= time < task.end:
                transit += task.vehicles
    return idle, local, transit


def feasible(initial, tasks):
    if not all(valid_task(task) for task in tasks):
        return False
    event_times = {0} | {t for task in tasks for t in (task.start, task.end)}
    for time in event_times:
        idle, local, transit = state(initial, tasks, time)
        if min(idle.values()) < 0:
            return False
        assert sum(idle.values()) + local + transit == sum(initial.values())
    return True


def terminal_target(initial, tasks, time, target):
    """True closing only; ongoing service must not use this helper."""
    idle, local, transit = state(initial, tasks, time)
    return (all(task.end <= time for task in tasks)
            and feasible(initial, tasks) and local == transit == 0
            and idle == target)


def main():
    initial = {1: 1, 2: 0}
    loaded = Task("loaded", 1, 2, 0, 2, cargo=1)
    empty = Task("empty", 2, 1, 3, 5)
    cases = []

    def check(name, condition):
        assert condition, name
        cases.append(name)
        print("PASS", name)

    check("nonbalanced_window_and_in_transit_are_legal",
          feasible(initial, [loaded])
          and state(initial, [loaded], 1) == ({1: 0, 2: 0}, 0, 1)
          and state(initial, [loaded], 2) == ({1: 0, 2: 1}, 0, 0))
    check("empty_reposition_without_any_orders",
          feasible(initial, [Task("empty", 1, 2, 0, 2)])
          and state(initial, [Task("empty", 1, 2, 0, 2)], 2)[0][2] == 1)
    check("in_transit_vehicle_cannot_be_reused",
          not feasible(initial, [loaded, Task("local", 2, 2, 1, 3)]))
    after_arrival = Task("local", 2, 2, 2, 3)
    check("completed_migration_survives_next_window",
          feasible(initial, [loaded, after_arrival])
          and not feasible(initial, [after_arrival]))
    check("true_terminal_target_counts_loaded_plus_empty",
          terminal_target(initial, [loaded, empty], 5, initial)
          and not terminal_target(initial, [loaded], 5, initial)
          and not terminal_target(initial, [loaded, empty], 4, initial))
    rebased = state(initial, [loaded], 2)[0]
    check("absolute_and_rebased_history_must_not_be_combined",
          feasible(rebased, [after_arrival])
          and not feasible(rebased, [loaded, after_arrival]))
    check("empty_cannot_carry_or_serve_cargo",
          not valid_task(Task("empty", 1, 2, 0, 2, cargo=0.25))
          and sum(task.cargo for task in [empty]) == 0)
    check("empty_duration_is_positive_fixed_tau",
          not valid_task(Task("empty", 1, 2, 0, 0))
          and not valid_task(Task("empty", 1, 2, 0, 3)))
    check("capacity_upper_bound_allows_zero_loaded_cargo",
          valid_task(Task("loaded", 1, 2, 0, 2, cargo=0)))
    check("true_closing_rejects_tasks_scheduled_after_terminal",
          not terminal_target(initial, [Task("empty", 1, 2, 6, 8)],
                              5, initial))
    print(f"{len(cases)} cases passed; deterministic ledger only; no MILP claim.")


if __name__ == "__main__":
    main()
