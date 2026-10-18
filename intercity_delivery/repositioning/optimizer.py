"""Opt-in V0-F MILP with a separate, order-independent empty vehicle network.

The inherited cargo model is deliberately unchanged: this is not the V1 wave
formulation. All event times are absolute and completed moves remain in history.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from intercity_delivery.models.flexible_direct_optimizer import FlexibleDirectOptimizer
from .config import RepositioningPolicy


class RepositioningOptimizer(FlexibleDirectOptimizer):
    model_id = "v0_f_explicit_empty_v1"
    vehicle_groups = frozenset(("x_manual", "y_auto", "w_direct", "empty_vehicles"))

    def __init__(self, config, data, policy=None):
        self.policy = (policy or RepositioningPolicy()).resolve(config)
        super().__init__(config, data)
        self.empty_arcs = [
            (i, i + self.policy.empty_travel_time_periods)
            for i in range(config.T)
            if i + self.policy.empty_travel_time_periods <= config.T
        ]

    def setup_variables(self):
        super().setup_variables()
        self.empty_vehicles = self.model.addVars(
            self.empty_arcs, self.flow, vtype=GRB.INTEGER, lb=0,
            name="empty_vehicles",
        )

    def set_objective(self):
        super().set_objective()
        self.cost_empty = gp.quicksum(
            self.policy.empty_cost_per_hour * self.cfg.period_hours * (j - i) * variable
            for (i, j, _flow), variable in self.empty_vehicles.items()
        )
        self.model.setObjective(
            self.penalty_unserved + self.cost_manual + self.cost_auto
            + self.cost_direct + self.cost_transfer + self.cost_empty,
            GRB.MINIMIZE,
        )

    def _city_idle(self, city, t):
        outgoing = "+" if city == 1 else "-"
        incoming = "-" if city == 1 else "+"
        movements = (self.w_direct, self.empty_vehicles)
        departures = gp.quicksum(
            variable for group in movements for (i, _j, flow), variable in group.items()
            if flow == outgoing and i <= t
        )
        arrivals = gp.quicksum(
            variable for group in movements for (_i, j, flow), variable in group.items()
            if flow == incoming and j <= t
        )
        busy_local = gp.quicksum(
            variable for (i, j, k, _flow), variable in self.x_manual.items()
            if k == city and i <= t < j
        )
        return self.cfg.N_manual[city] - departures + arrivals - busy_local

    def _add_manual_fleet_constraints(self):
        # Both cities, including the true horizon endpoint. No window reset.
        for t in range(self.cfg.T + 1):
            for city in (1, 2):
                self.model.addConstr(
                    self._city_idle(city, t) >= 0,
                    name=f"RE_ManualInventory_{city}_{t}",
                )
        if self.policy.terminal_target is not None:
            for city in (1, 2):
                self.model.addConstr(
                    self._city_idle(city, self.cfg.T) == self.policy.terminal_target[city],
                    name=f"RE_RealTerminalTarget_{city}_{self.cfg.T}",
                )

    def decision_variable_groups(self):
        return {**super().decision_variable_groups(), "empty_vehicles": self.empty_vehicles}

    def configure_event_window(self, current_time, start_end, completion_end, committed):
        """Fix the whole past exactly; limit new origin starts and completion.

        Auto and destination operations may start in the extension, so a pickup
        committed in an earlier window can finish its cargo chain. New operations
        must finish by completion_end; existing commitments are never truncated.
        """
        if not 0 <= current_time < start_end <= completion_end <= self.cfg.T:
            raise ValueError("Invalid start/completion event window")
        if self.policy.terminal_target is not None:
            raise ValueError(
                "Rolling terminal_target is not supported: a tail reachability "
                "certificate to the real T is required. Use full-horizon mode."
            )
        groups = self.decision_variable_groups()
        for name, history in committed.items():
            if name not in groups:
                raise ValueError(f"Unknown committed group: {name}")
            for key, value in history.items():
                if abs(value) > 1e-8 and tuple(key) not in groups[name]:
                    raise ValueError(f"Committed event missing from model: {name} {key}")
        for name, variables in groups.items():
            for key, variable in variables.items():
                i, j = key[:2]
                if i < current_time:
                    value = float(committed.get(name, {}).get(tuple(key), 0.0))
                    if name in self.vehicle_groups:
                        if abs(value - round(value)) > 1e-6:
                            raise ValueError("Committed vehicle count must be integral")
                        value = float(round(value))
                    variable.lb = value
                    variable.ub = value
                    continue
                start_limit = completion_end
                if name in ("w_direct", "h_direct", "empty_vehicles"):
                    start_limit = start_end
                elif name == "g_manual":
                    origin = 1 if key[3] == "+" else 2
                    if key[2] == origin:
                        start_limit = start_end
                if i >= start_limit or j > completion_end:
                    variable.ub = 0.0
        self.model.update()

    def extract_decisions(self):
        # Keep small continuous loads. Reporting display thresholds are not a
        # commitment or accounting tolerance.
        result = {}
        for name, group in self.decision_variable_groups().items():
            result[name] = {}
            for key, variable in group.items():
                value = float(variable.X)
                if name in self.vehicle_groups:
                    if abs(value - round(value)) > 1e-6:
                        raise ValueError("Solver incumbent has nonintegral vehicles")
                    value = float(round(value))
                if abs(value) > 1e-8:
                    result[name][tuple(key)] = value
        return result
