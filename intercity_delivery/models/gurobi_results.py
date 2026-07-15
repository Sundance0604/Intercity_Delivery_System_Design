"""Safe accessors for Gurobi attributes that are status/model dependent."""

from typing import Optional

import gurobipy as gp


def optional_model_float(model: gp.Model, attribute: str) -> Optional[float]:
    """Return a numeric model attribute, or ``None`` when it is unavailable.

    Attributes such as ``MIPGap`` are undefined for a continuous reduced
    window even when that window has a valid solution.  Gurobi reports that
    situation by raising ``AttributeError``/``GurobiError`` from ``getAttr``.
    """

    try:
        return float(model.getAttr(attribute))
    except (AttributeError, gp.GurobiError, TypeError, ValueError):
        return None
