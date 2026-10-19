"""V1-F arc-wave MILP with independent empty moves and absolute inventories.

The local selector uses the global fleet (a disclosed migration correction).
No computational window imposes a city return target.
"""

import math
from collections import defaultdict

from .capacity import _integer, local_capacity
from .flexible_capacity import direct_arcs, direct_capacity, empty_arcs
from .flexible_configuration import empty_duration, empty_hourly_cost, validate_f_instance
from .network import av_arcs, local_arcs


class V1FlexibleOptimizer:
    model_id = "v1_flexible_arc_wave_empty_v1"
    selector_bound = "global_manual_fleet"

    def __init__(self, config, orders, profile, candidates=None):
        validate_f_instance(config, orders, profile)
        if candidates is not None and (not isinstance(candidates, dict) or
                set(candidates)-{"g_manual", "g_auto", "h_direct"}):
            raise ValueError("candidates may filter only cargo groups")
        self.config, self.orders, self.profile = config, tuple(orders), profile
        self.candidates = candidates
        self.model = None
        self.arcs_manual = {k: local_arcs(config.T, config.t0_minutes, config.a[k],
                                        config.b[k], config.rho[k], config.M)
                            for k in (1, 2)}
        self.arcs_auto = av_arcs(config.T, config.tau_periods)
        self.arcs_direct, self.arcs_empty = direct_arcs(config), empty_arcs(config)
        self.manual_fleet = sum(config.N_manual.values())
        self.capacities = {(i, j, k, v): local_capacity(
            (j-i)*config.t0_minutes, v, config.a[k], config.b[k], config.rho[k], config.M)
            for k in (1, 2) for i, j in self.arcs_manual[k]
            for v in range(1, self.manual_fleet+1)}
        self.direct_capacities = {(i, j, flow, v): direct_capacity(config, i, j, flow, v)
                                 for i, j in self.arcs_direct for flow in ("+", "-")
                                 for v in range(1, self.manual_fleet+1)}

    def _allowed(self, group, keys):
        if self.candidates is None or group not in self.candidates:
            return keys
        return [key for key in keys if key in self.candidates[group]]

    def build_model(self):
        if self.model is not None:
            raise RuntimeError("optimizer has already been built")
        import gurobipy as gp
        from gurobipy import GRB

        cfg, fleet = self.config, self.manual_fleet
        self.model = model = gp.Model(self.model_id)
        try:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.Seed = 0
            model.Params.FeasibilityTol = 1e-8
            model.Params.IntFeasTol = 1e-8
            xkeys = [(i, j, k, f) for k in (1, 2) for i, j in self.arcs_manual[k]
                     for f in ("+", "-")]
            ykeys = [(i, j, f) for i, j in self.arcs_auto for f in ("+", "-")]
            wkeys = [(i, j, f) for i, j in self.arcs_direct for f in ("+", "-")]
            ekeys = [(i, j, f) for i, j in self.arcs_empty for f in ("+", "-")]
            gkeys = self._allowed("g_manual", [(i, j, k, o.flow, o.order_id)
                for o in self.orders for k in (1, 2) for i, j in self.arcs_manual[k]])
            akeys = self._allowed("g_auto", [(i, j, o.flow, o.order_id)
                for o in self.orders for i, j in self.arcs_auto])
            hkeys = self._allowed("h_direct", [(i, j, o.flow, o.order_id)
                for o in self.orders for i, j in self.arcs_direct])
            definitions = {
                "x_manual": (xkeys, GRB.INTEGER), "y_auto": (ykeys, GRB.INTEGER),
                "w_direct": (wkeys, GRB.INTEGER), "empty_vehicles": (ekeys, GRB.INTEGER),
                "u_wave": ([(*key, v) for key in xkeys for v in range(1, fleet+1)], GRB.BINARY),
                "u_direct": ([(*key, v) for key in wkeys for v in range(1, fleet+1)], GRB.BINARY),
                "g_manual": (gkeys, GRB.CONTINUOUS), "g_auto": (akeys, GRB.CONTINUOUS),
                "h_direct": (hkeys, GRB.CONTINUOUS)}
            self.groups = {}
            for name, (keys, domain) in definitions.items():
                variables = model.addVars(keys, vtype=domain, lb=0, name=name)
                setattr(self, name, variables)
                self.groups[name] = variables
            # Preserve the same mother domains for dense/pruned comparisons.
            # Mechanisms constrain vehicle and cargo variables, not cargo ratios.
            self._apply_mechanism_bounds()
            self.z_unserved = {(o.order_id,): model.addVar(lb=0, ub=o.quantity,
                                  name=f"z_unserved_{n}") for n, o in enumerate(self.orders)}
            self.groups["z_unserved"] = self.z_unserved

            # Sparse indices avoid indexing any cargo key eliminated by pruning.
            local_cargo, auto_cargo, direct_cargo = defaultdict(list), defaultdict(list), defaultdict(list)
            order_local, order_auto, order_direct = defaultdict(list), defaultdict(list), defaultdict(list)
            for key, var in self.g_manual.items():
                local_cargo[key[:4]].append(var)
                order_local[key[-1], key[2]].append((key, var))
            for key, var in self.g_auto.items():
                auto_cargo[key[:3]].append(var)
                order_auto[key[-1]].append((key, var))
            for key, var in self.h_direct.items():
                direct_cargo[key[:3]].append(var)
                order_direct[key[-1]].append((key, var))

            cost = lambda group, rate: gp.quicksum(
                var*(key[1]-key[0])*cfg.t0_minutes/60*rate for key, var in group.items())
            self.cost_manual = cost(self.x_manual, cfg.c_manual_hour)
            self.cost_auto = cost(self.y_auto, cfg.c_auto_hour)
            self.cost_direct = cost(self.w_direct, cfg.c_direct_hour)
            self.cost_empty = cost(self.empty_vehicles, empty_hourly_cost(cfg))
            self.cost_unserved = gp.quicksum(o.penalty*self.z_unserved[o.order_id,] for o in self.orders)
            self.cost_transfer = gp.quicksum(var*cfg.transfer_cost_per_unit
                for o in self.orders for key, var in order_local[o.order_id, 2 if o.flow == "+" else 1])
            model.setObjective(self.cost_manual+self.cost_auto+self.cost_direct+self.cost_empty+
                               self.cost_unserved+self.cost_transfer, GRB.MINIMIZE)

            for t in range(cfg.T+1):
                for city, outgoing in ((1, "+"), (2, "-")):
                    incoming = "-" if outgoing == "+" else "+"
                    stock = cfg.N_manual[city]-gp.quicksum(
                        var for group in (self.w_direct, self.empty_vehicles)
                        for (i, j, f), var in group.items() if f == outgoing and i <= t)
                    stock += gp.quicksum(var for group in (self.w_direct, self.empty_vehicles)
                        for (i, j, f), var in group.items() if f == incoming and j <= t)
                    stock -= gp.quicksum(var for (i, j, k, f), var in self.x_manual.items()
                                        if k == city and i <= t < j)
                    model.addConstr(stock >= 0, name=f"manual_stock_{city}_{t}")
                    auto_stock = cfg.N_auto[city]-gp.quicksum(var for (i, j, f), var in self.y_auto.items()
                        if f == outgoing and i <= t)+gp.quicksum(var for (i, j, f), var in self.y_auto.items()
                        if f == incoming and j <= t)
                    model.addConstr(auto_stock >= 0, name=f"auto_stock_{city}_{t}")
                model.addConstr(gp.quicksum(var for (i, j, f), var in self.y_auto.items()
                    if i <= t < j) <= sum(cfg.N_auto.values()), name=f"auto_busy_{t}")

            for key in xkeys:
                i, j, city, flow = key
                model.addConstr(self.x_manual[key] == gp.quicksum(v*self.u_wave[*key, v]
                    for v in range(1, fleet+1)), name=f"local_link_{i}_{j}_{city}_{flow}")
                model.addConstr(gp.quicksum(self.u_wave[*key, v] for v in range(1, fleet+1)) <= 1)
                model.addConstr(gp.quicksum(local_cargo[key]) <= gp.quicksum(
                    self.capacities[i, j, city, v]*self.u_wave[*key, v]
                    for v in range(1, fleet+1)))
            for key in wkeys:
                model.addConstr(self.w_direct[key] == gp.quicksum(v*self.u_direct[*key, v]
                    for v in range(1, fleet+1)))
                model.addConstr(gp.quicksum(self.u_direct[*key, v] for v in range(1, fleet+1)) <= 1)
                if cfg.min_direct_load_per_vehicle > 0:
                    model.addConstr(gp.quicksum(direct_cargo[key]) >=
                                    cfg.min_direct_load_per_vehicle*self.w_direct[key],
                                    name=f"direct_minimum_load_{key}")
                model.addConstr(gp.quicksum(direct_cargo[key]) <= gp.quicksum(
                    self.direct_capacities[*key, v]*self.u_direct[*key, v]
                    for v in range(1, fleet+1)))
            for key in ykeys:
                model.addConstr(gp.quicksum(auto_cargo[key]) <= cfg.M_auto*self.y_auto[key])

            for o in self.orders:
                oid = o.order_id
                origin = 1 if o.flow == "+" else 2
                dest = 3-origin
                for city in (1, 2):
                    entries = order_local[oid, city]
                    for key, var in entries:
                        if ((city == origin and key[0] < o.earliest_start) or
                                (city == dest and key[1] > o.latest_completion)):
                            var.UB = 0
                    model.addConstr(gp.quicksum(var for _, var in entries)+
                        gp.quicksum(var for _, var in order_direct[oid])+self.z_unserved[oid,]
                        == o.quantity)
                for key, var in order_direct[oid]:
                    if key[0] < o.earliest_start or key[1] > o.latest_completion:
                        var.UB = 0
                for t in range(cfg.T+1):
                    model.addConstr(gp.quicksum(var for key, var in order_auto[oid] if key[0] <= t)
                        <= gp.quicksum(var for key, var in order_local[oid, origin]
                                       if key[1]+cfg.transfer_time_periods <= t))
                    model.addConstr(gp.quicksum(var for key, var in order_local[oid, dest] if key[0] <= t)
                        <= gp.quicksum(var for key, var in order_auto[oid]
                                       if key[1]+cfg.transfer_time_periods <= t))
            direct_total = gp.quicksum(self.h_direct.values())
            served_total = gp.quicksum(o.quantity-self.z_unserved[o.order_id,] for o in self.orders)
            model.addConstr(direct_total >= cfg.direct_ratio_min*served_total)
            model.addConstr(direct_total <= cfg.direct_ratio_max*served_total)
            if cfg.terminal_target is not None:
                for city, outgoing in ((1, "+"), (2, "-")):
                    incoming = "-" if outgoing == "+" else "+"
                    final = cfg.N_manual[city]-gp.quicksum(var
                        for group in (self.w_direct, self.empty_vehicles)
                        for (i, j, f), var in group.items() if f == outgoing)
                    final += gp.quicksum(var for group in (self.w_direct, self.empty_vehicles)
                        for (i, j, f), var in group.items() if f == incoming)
                    model.addConstr(final == cfg.terminal_target[city], name=f"true_terminal_{city}")
            model.update()
            return self
        except Exception:
            model.dispose()
            self.model = None
            raise

    def _apply_mechanism_bounds(self):
        disabled = []
        if self.config.service_mechanism in ("T0", "T1"):
            disabled.extend(("w_direct", "u_direct", "h_direct"))
        if self.config.service_mechanism in ("T0", "T2"):
            disabled.append("empty_vehicles")
        for name in disabled:
            for variable in self.groups[name].values():
                variable.UB = 0

    def configure_window(self, current_time, start_end, completion_end, committed):
        """Keep all past starts exactly; only new tasks obey this window's cut."""
        cfg = self.config
        if self.model is None:
            raise RuntimeError("build_model must be called first")
        if cfg.terminal_target is not None:
            raise ValueError("rolling terminal_target requires unimplemented tail reachability")
        for name, value in (("current_time", current_time), ("start_end", start_end),
                            ("completion_end", completion_end)):
            _integer(name, value)
        if not 0 <= current_time <= start_end <= completion_end <= cfg.T:
            raise ValueError("invalid rolling window boundaries")
        if not isinstance(committed, dict) or set(committed)-set(self.groups):
            raise ValueError("unknown committed variable group")
        for name, values in committed.items():
            if not isinstance(values, dict):
                raise ValueError("committed groups must be dictionaries")
            for key, value in values.items():
                if (isinstance(value, bool) or not isinstance(value, (float, int)) or
                        not math.isfinite(value) or value < 0):
                    raise ValueError("committed values must be finite nonnegative numbers")
                if name == "z_unserved" and value != 0:
                    raise ValueError("unserved quantities are not committed events")
                if value != 0 and (key not in self.groups[name] or key[0] >= current_time):
                    raise ValueError("nonzero committed event is absent or has not started")
        for name, group in self.groups.items():
            if name == "z_unserved":
                continue
            history = committed.get(name, {})
            for key, var in group.items():
                i, j = key[:2]
                if i < current_time:
                    value = history.get(key, 0.0)
                    if value > var.UB+1e-7 or value < var.LB-1e-7:
                        raise ValueError("committed event conflicts with model bounds")
                    var.LB = var.UB = value
                    continue
                cutoff = start_end
                if name in ("y_auto", "g_auto"):
                    cutoff = completion_end
                elif name in ("x_manual", "u_wave", "g_manual"):
                    city, flow = key[2:4]
                    if city != (1 if flow == "+" else 2):
                        cutoff = completion_end
                if i >= cutoff or j > completion_end:
                    var.UB = 0
        self.model.update()
        return self

    def extract_schedule(self):
        if self.model is None or self.model.SolCount == 0:
            raise ValueError("no incumbent is available")
        return {name: {tuple(key): float(var.X) for key, var in group.items()}
                for name, group in self.groups.items()}
