"""Isolated full-horizon V1-M MILP under the adopted D1--D6-A profile.

No V0 classes, direct transportation, empty manual repositioning, rolling
horizon, steady-state model, or QVI is used. The required profile defines
the adopted benchmark semantics; the manuscript itself remains unchanged.
"""

from .capacity import local_capacity
from .configuration import PROFILE_SEMANTIC_STATUS, validate_instance
from .network import av_arcs, local_arcs


class V1MandatoryOptimizer:
    model_id = "v1_mandatory_arc_wave_v1"
    semantic_status = PROFILE_SEMANTIC_STATUS
    traceability = {
        "M01": "main_v2_BHH.tex:74", "M02": "main_v2_BHH.tex:82",
        "M03": "eq:num man veh", "M04": "num auto veh",
        "M05": "bal auto +; bal auto -", "M06": "BHH",
        "M07": "wave link", "M08": "auto vol", "M09": "violate; E",
        "M10": "transfer step 1", "M11": "transfer step 2",
        "M12": "unserved", "M13": "domain", "M14": "eq:obj",
        "M15": "implementation requirement: complete export",
    }

    def __init__(self, config, orders, profile):
        # Validate before a Gurobi model/environment is created.
        validate_instance(config, orders, profile)
        self.config = config
        self.orders = tuple(orders)
        self.profile = profile
        self.model = None
        self.arcs_manual = {
            city: local_arcs(config.T, config.t0_minutes, config.a[city],
                             config.b[city], config.rho[city], config.M)
            for city in (1, 2)
        }
        self.arcs_auto = av_arcs(config.T, config.tau_periods)
        self.capacities = {
            (i, j, city, v): local_capacity(
                (j-i)*config.t0_minutes, v, config.a[city], config.b[city],
                config.rho[city], config.M)
            for city in (1, 2) for i, j in self.arcs_manual[city]
            for v in range(1, config.N_manual[city]+1)
        }

    def build_model(self):
        """Build every M01--M15 structure; do not optimize or infer a profile."""
        if self.model is not None:
            raise RuntimeError("this optimizer has already been built")
        # Lazy import preserves pure configuration/kernel use without a solver.
        import gurobipy as gp
        from gurobipy import GRB

        cfg = self.config
        model = self.model = gp.Model(self.model_id)
        try:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.Seed = 0
            model.Params.FeasibilityTol = 1e-8
            model.Params.IntFeasTol = 1e-8
            x_keys = [(i, j, city, flow) for city in (1, 2)
                      for i, j in self.arcs_manual[city] for flow in ("+", "-")]
            y_keys = [(i, j, flow) for i, j in self.arcs_auto for flow in ("+", "-")]
            u_keys = [(*key, v) for key in x_keys
                      for v in range(1, cfg.N_manual[key[2]]+1)]
            g_keys = [(i, j, city, order.flow, order.order_id)
                      for order in self.orders for city in (1, 2)
                      for i, j in self.arcs_manual[city]]
            ga_keys = [(i, j, order.flow, order.order_id)
                       for order in self.orders for i, j in self.arcs_auto]
            self.x_manual = model.addVars(x_keys, vtype=GRB.INTEGER, lb=0, name="x_manual")
            self.y_auto = model.addVars(y_keys, vtype=GRB.INTEGER, lb=0, name="y_auto")
            self.u_wave = model.addVars(u_keys, vtype=GRB.BINARY, name="u_wave")
            self.g_manual = model.addVars(g_keys, lb=0, name="g_manual")
            self.g_auto = model.addVars(ga_keys, lb=0, name="g_auto")
            # Explicit one-element tuples avoid Gurobi's single-index flattening.
            self.z_unserved = {(order.order_id,): model.addVar(
                lb=0, ub=order.quantity, name=f"z_unserved_{position}")
                for position, order in enumerate(self.orders)}
            self.groups = {name: getattr(self, name) for name in (
                "x_manual", "y_auto", "u_wave", "g_manual", "g_auto", "z_unserved")}

            # M14: all times are minutes and both vehicle rates are per hour.
            self.cost_manual = gp.quicksum(
                variable*(j-i)*cfg.t0_minutes/60*cfg.c_manual_hour
                for (i, j, city, flow), variable in self.x_manual.items())
            self.cost_auto = gp.quicksum(
                variable*cfg.tau_periods*cfg.t0_minutes/60*cfg.c_auto_hour
                for variable in self.y_auto.values())
            self.cost_unserved = gp.quicksum(
                order.penalty*self.z_unserved[order.order_id,] for order in self.orders)
            model.setObjective(self.cost_manual+self.cost_auto+self.cost_unserved,
                               GRB.MINIMIZE)

            # M03--M05: half-open occupation and absolute two-hub AV stocks.
            for t in range(cfg.T+1):
                for city in (1, 2):
                    model.addConstr(gp.quicksum(
                        variable for (i, j, k, flow), variable in self.x_manual.items()
                        if k == city and i <= t < j) <= cfg.N_manual[city],
                        name=f"M03_manual_{city}_{t}")
                model.addConstr(gp.quicksum(
                    variable for (i, j, flow), variable in self.y_auto.items()
                    if i <= t < j) <= sum(cfg.N_auto.values()), name=f"M04_auto_{t}")
                for city, outgoing, incoming in ((1, "+", "-"), (2, "-", "+")):
                    stock = cfg.N_auto[city]-gp.quicksum(
                        variable for (i, j, flow), variable in self.y_auto.items()
                        if flow == outgoing and i <= t)+gp.quicksum(
                        variable for (i, j, flow), variable in self.y_auto.items()
                        if flow == incoming and j <= t)
                    model.addConstr(stock >= 0, name=f"M05_hub_{city}_{t}")

            for i, j, city, flow in x_keys:
                selected = [self.u_wave[i, j, city, flow, v]
                            for v in range(1, cfg.N_manual[city]+1)]
                model.addConstr(self.x_manual[i, j, city, flow] == gp.quicksum(
                    v*self.u_wave[i, j, city, flow, v]
                    for v in range(1, cfg.N_manual[city]+1)),
                    name=f"M07_link_{i}_{j}_{city}_{flow}")
                model.addConstr(gp.quicksum(selected) <= 1,
                                name=f"M07_onehot_{i}_{j}_{city}_{flow}")
                model.addConstr(gp.quicksum(
                    self.g_manual[i, j, city, flow, order.order_id]
                    for order in self.orders if order.flow == flow) <= gp.quicksum(
                        self.capacities[i, j, city, v]*self.u_wave[i, j, city, flow, v]
                        for v in range(1, cfg.N_manual[city]+1)),
                    name=f"M06_capacity_{i}_{j}_{city}_{flow}")
            for i, j, flow in y_keys:
                model.addConstr(gp.quicksum(
                    self.g_auto[i, j, flow, order.order_id]
                    for order in self.orders if order.flow == flow)
                    <= cfg.M_auto*self.y_auto[i, j, flow],
                    name=f"M08_shared_{i}_{j}_{flow}")

            for position, order in enumerate(self.orders):
                oid, flow = order.order_id, order.flow
                origin = 1 if flow == "+" else 2
                destination = 3-origin
                # M09: apply only the specified leg's time-window condition.
                for city in (1, 2):
                    for i, j in self.arcs_manual[city]:
                        if ((city == origin and i < order.earliest_start)
                                or (city == destination and j > order.latest_completion)):
                            model.addConstr(self.g_manual[i, j, city, flow, oid] == 0,
                                            name=f"M09_window_{position}_{city}_{i}_{j}")
                    model.addConstr(gp.quicksum(
                        self.g_manual[i, j, city, flow, oid]
                        for i, j in self.arcs_manual[city])+self.z_unserved[oid,]
                        == order.quantity, name=f"M12_demand_{position}_{city}")
                for t in range(cfg.T+1):
                    model.addConstr(gp.quicksum(
                        self.g_auto[i, j, flow, oid] for i, j in self.arcs_auto if i <= t)
                        <= gp.quicksum(self.g_manual[i, j, origin, flow, oid]
                                       for i, j in self.arcs_manual[origin] if j <= t),
                        name=f"M10_pickup_before_AV_{position}_{t}")
                    model.addConstr(gp.quicksum(
                        self.g_manual[i, j, destination, flow, oid]
                        for i, j in self.arcs_manual[destination] if i <= t)
                        <= gp.quicksum(self.g_auto[i, j, flow, oid]
                                       for i, j in self.arcs_auto if j <= t),
                        name=f"M11_AV_before_delivery_{position}_{t}")
            model.update()
            return self
        except Exception:
            model.dispose()
            self.model = None
            raise

    def extract_schedule(self):
        """Export every variable, including zero and sub-0.1 continuous flow."""
        if self.model is None or self.model.SolCount == 0:
            raise ValueError("no incumbent is available")
        return {name: {tuple(key): float(variable.X) for key, variable in group.items()}
                for name, group in self.groups.items()}
