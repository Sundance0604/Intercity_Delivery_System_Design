"""Independent absolute-time fleet/accounting checks, not a full cargo validator.

Completed migrations stay in the ledger. Do not combine rebased initial fleets
with these absolute historical events. Arrivals at t are available at t.
"""

from dataclasses import dataclass
from .config import RepositioningPolicy, integer, nonnegative, fleet


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    origin: int
    destination: int
    start: int
    end: int
    vehicles: int = 1
    cargo: float = 0.0

    def validate(self, empty_travel_time_periods):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("Task requires a nonempty unique string ID")
        if self.kind not in {"local", "loaded", "empty"}:
            raise ValueError("Unknown task kind")
        if type(self.origin) is not int or self.origin not in (1, 2):
            raise ValueError("Invalid origin")
        if type(self.destination) is not int or self.destination not in (1, 2):
            raise ValueError("Invalid destination")
        integer(self.start, "start")
        integer(self.end, "end", 1)
        integer(self.vehicles, "vehicles", 1)
        nonnegative(self.cargo, "cargo")
        if self.end <= self.start:
            raise ValueError("Task duration must be positive")
        if (self.origin == self.destination) != (self.kind == "local"):
            raise ValueError("Only local tasks may have equal cities")
        if self.kind == "empty":
            integer(empty_travel_time_periods, "empty_travel_time_periods", 1)
            if self.end - self.start != empty_travel_time_periods or self.cargo != 0:
                raise ValueError("Empty tasks have fixed physical duration and zero cargo")
        return self


def validate_tasks(initial, tasks, empty_travel_time_periods):
    fleet(initial, "initial fleet")
    seen = set()
    for task in tasks:
        task.validate(empty_travel_time_periods)
        if task.task_id in seen:
            raise ValueError(f"Duplicate task ID: {task.task_id}")
        seen.add(task.task_id)


def state(initial, tasks, time):
    integer(time, "time")
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
    return {"idle": idle, "local_busy": local, "in_transit": transit}


def validate_ledger(initial, tasks, empty_travel_time_periods):
    tasks = list(tasks)
    validate_tasks(initial, tasks, empty_travel_time_periods)
    for time in sorted({0} | {t for task in tasks for t in (task.start, task.end)}):
        snapshot = state(initial, tasks, time)
        if min(snapshot["idle"].values()) < 0:
            raise ValueError(f"Vehicle inventory negative at {time}")
        if sum(snapshot["idle"].values()) + snapshot["local_busy"] + snapshot["in_transit"] != sum(initial.values()):
            raise ValueError("Vehicle conservation failed")
    return True


def check_terminal_target(initial, tasks, time, target, empty_travel_time_periods):
    tasks = list(tasks)
    fleet(target, "terminal target")
    validate_ledger(initial, tasks, empty_travel_time_periods)
    snapshot = state(initial, tasks, time)
    return (all(task.end <= time for task in tasks)
            and snapshot["local_busy"] == snapshot["in_transit"] == 0
            and snapshot["idle"] == target)


VEHICLE_GROUPS = {"x_manual": 4, "y_auto": 3, "w_direct": 3, "empty_vehicles": 3}
FLOW_GROUPS = {"g_manual": 5, "g_auto": 4, "h_direct": 4}
SUMMARY_GROUPS = {"r_transshipment", "q_direct", "z_unserved"}


def _decisions_checked(decisions):
    result = {}
    for group, values in decisions.items():
        if group in SUMMARY_GROUPS:
            # Completion is reconstructed independently; model r/q/z are plans.
            continue
        size = VEHICLE_GROUPS.get(group, FLOW_GROUPS.get(group))
        if size is None:
            raise ValueError(f"Unknown decision group: {group}")
        result[group] = {}
        for key, value in values.items():
            if not isinstance(key, tuple) or len(key) != size:
                raise ValueError(f"Invalid {group} key")
            i, j = key[:2]
            integer(i, "arc start")
            integer(j, "arc end", 1)
            if i >= j:
                raise ValueError("Arc must have positive duration")
            flow_index = 3 if group in ("x_manual", "g_manual") else 2
            if key[flow_index] not in ("+", "-"):
                raise ValueError("Invalid flow direction")
            if flow_index == 3 and (type(key[2]) is not int or key[2] not in (1, 2)):
                raise ValueError("Invalid local city")
            nonnegative(value, group)
            if group in VEHICLE_GROUPS:
                # Solver exports exact integral floats. Nonintegral floats and
                # bools are rejected rather than silently rounded here.
                if isinstance(value, bool) or int(value) != value:
                    raise ValueError("Vehicle decisions must be integral")
                value = int(value)
            if value != 0:
                result[group][key] = value
    return result


def tasks_from_decisions(decisions, empty_travel_time_periods):
    decisions = _decisions_checked(decisions)
    manual, auto = [], []
    for group in VEHICLE_GROUPS:
        for key, vehicles in decisions.get(group, {}).items():
            i, j = key[:2]
            if group == "x_manual":
                city = key[2]
                origin = destination = city
                kind = "local"
            else:
                origin = 1 if key[2] == "+" else 2
                destination = 3 - origin
                kind = "empty" if group == "empty_vehicles" else "loaded"
            cargo_group = "g_manual" if group == "x_manual" else ("g_auto" if group == "y_auto" else "h_direct")
            cargo = (0.0 if group == "empty_vehicles" else
                     sum(value for load_key, value in decisions.get(cargo_group, {}).items()
                         if load_key[:-1] == key))
            task = Task(f"{group}:{key!r}", kind, origin, destination, i, j, vehicles, cargo)
            task.validate(empty_travel_time_periods)
            (auto if group == "y_auto" else manual).append(task)
    return manual, auto


def evaluate_committed(config, orders, decisions, resolvedpolicy=None,
                       completed_horizon=None, completed=False):
    """Reconstruct billed commitments and cargo actually delivered by the cut.

    Entire duration of an initiated commitment is billed once, even if still in
    flight. Future planned departures are disallowed. Partial totals are not
    complete-run objectives. Cargo causality, BHH capacity, service windows and
    direct-share policy are explicitly outside this validator's scope.
    """
    policy = (resolvedpolicy or RepositioningPolicy()).resolve(config)
    integer(config.travel_time_periods, "travel_time_periods", 1)
    if not isinstance(completed, bool):
        raise ValueError("completed must be bool")
    cut = config.T if completed_horizon is None else integer(completed_horizon, "completed_horizon")
    if cut > config.T or (completed and cut != config.T):
        raise ValueError("Completion status and observation horizon disagree")
    orders = getattr(orders, "all_orders", orders)
    decisions = _decisions_checked(decisions)
    manual, auto = tasks_from_decisions(decisions, policy.empty_travel_time_periods)
    for group, values in decisions.items():
        for key in values:
            if key[0] >= cut or key[1] > config.T:
                raise ValueError("Only initiated commitments within the true horizon may be evaluated")
    validate_ledger(config.N_manual, manual, policy.empty_travel_time_periods)
    validate_ledger(config.N_auto, auto, policy.empty_travel_time_periods)
    if any(task.end - task.start != config.travel_time_periods for task in auto):
        raise ValueError("Automatic task duration differs from physical travel time")
    for order in orders.values():
        nonnegative(order.quantity, "order.quantity")
        nonnegative(order.penalty_lost, "order.penalty_lost")
        if order.flow not in ("+", "-"):
            raise ValueError("Invalid order flow")
    served_direct = {key: 0.0 for key in orders}
    served_transfer = {key: 0.0 for key in orders}
    for group in FLOW_GROUPS:
        for key, value in decisions.get(group, {}).items():
            order_id = key[-1]
            if order_id not in orders:
                raise ValueError("Flow refers to unknown order")
            order = orders[order_id]
            flow_index = 3 if group == "g_manual" else 2
            if key[flow_index] != order.flow:
                raise ValueError("Flow direction differs from order")
            if key[1] > cut:
                continue
            if group == "h_direct":
                served_direct[order_id] += value
            elif group == "g_manual" and key[2] == (2 if order.flow == "+" else 1):
                served_transfer[order_id] += value
    total_unserved = penalty = 0.0
    served_by_order = {}
    unserved_by_order = {}
    penalized_unserved_by_order = {}
    for order_id, order in orders.items():
        served = served_direct[order_id] + served_transfer[order_id]
        if served > order.quantity + 1e-6:
            raise ValueError("Delivered quantity exceeds demand")
        residual = max(0.0, order.quantity - served)
        served_by_order[str(order_id)] = served
        unserved_by_order[str(order_id)] = residual
        if completed or order.latest_completion <= cut:
            total_unserved += residual
            penalty += residual * order.penalty_lost
            penalized_unserved_by_order[str(order_id)] = residual
    cost = {"manual": 0.0, "auto": 0.0, "direct": 0.0, "empty": 0.0,
            "transfer": 0.0, "unserved": penalty}
    rates = {"x_manual": ("manual", config.cost_manual), "y_auto": ("auto", config.cost_auto),
             "w_direct": ("direct", config.cost_direct),
             "empty_vehicles": ("empty", policy.empty_cost_per_hour)}
    for group, (component, rate) in rates.items():
        cost[component] = sum((key[1] - key[0]) * config.period_hours * rate * value
                              for key, value in decisions.get(group, {}).items())
    direct_quantity = sum(served_direct.values())
    transfer_quantity = sum(served_transfer.values())
    served_quantity = direct_quantity + transfer_quantity
    cost["transfer"] = transfer_quantity * config.transfer_cost_per_unit
    terminal_met = None
    if completed and policy.terminal_target is not None:
        terminal_met = check_terminal_target(config.N_manual, manual, config.T,
                                             policy.terminal_target, policy.empty_travel_time_periods)
        if not terminal_met:
            raise ValueError("Explicit true terminal target not met")
    return {
        "total_cost": sum(cost.values()) if completed else None,
        "committed_cost": sum(cost.values()), "cost_components": cost,
        "completed": completed, "completed_horizon": cut,
        "penalty_scope": "all demand" if completed else "expired demand only",
        "cost_scope": "full executed horizon" if completed else "committed actions plus expired-demand penalty",
        "served_quantity": served_quantity, "direct_quantity": direct_quantity,
        "transshipment_quantity": transfer_quantity, "unserved_quantity": total_unserved,
        "served_by_order": served_by_order, "unserved_by_order": unserved_by_order,
        "penalized_unserved_by_order": penalized_unserved_by_order,
        "direct_share": direct_quantity / served_quantity if served_quantity else 0.0,
        "empty_vehicle_trips": sum(task.vehicles for task in manual if task.kind == "empty"),
        "direct_service_vehicle_trips": sum(task.vehicles for task in manual if task.kind == "loaded"),
        "positive_cargo_direct_arc_count": sum(task.cargo > 0 for task in manual if task.kind == "loaded"),
        "ledger_valid": True, "validation_scope": "manual/auto event inventory, scalar schema and accounting only; cargo causality, capacities, windows and policy feasibility NOT validated",
        "terminal_state": state(config.N_manual, manual, cut),
        "auto_terminal_state": state(config.N_auto, auto, cut),
        "terminal_target_met": terminal_met, "policy": policy.metadata(),
    }


evaluate_execution = evaluate_committed
