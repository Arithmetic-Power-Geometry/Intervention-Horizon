"""Core mathematics for Intervention Horizon analysis.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

ND = NormalDist()

@dataclass(frozen=True)
class Action:
    name: str
    travel_mean: float
    travel_sd: float
    reliability: float = 1.0
    capacity: int = 1
    cost: float = 0.0

@dataclass(frozen=True)
class HorizonResult:
    viable_actions: tuple[str, ...]
    horizon: float
    opportunity_volume: float
    fragility: float
    status: str


def action_viability_probability(hazard_mean: float, hazard_sd: float, action: Action) -> float:
    """Probability that an intervention completes before the hazard, times action reliability."""
    mu = hazard_mean - action.travel_mean
    sd = math.sqrt(max(hazard_sd, 0.0) ** 2 + max(action.travel_sd, 0.0) ** 2)
    p_time = (1.0 if mu > 0 else 0.0) if sd == 0 else ND.cdf(mu / sd)
    return max(0.0, min(1.0, p_time * action.reliability))


def viable_actions(hazard_mean: float, hazard_sd: float, actions: Sequence[Action], epsilon: float = 0.05) -> tuple[Action, ...]:
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    threshold = 1.0 - epsilon
    return tuple(a for a in actions if action_viability_probability(hazard_mean, hazard_sd, a) >= threshold)


def action_last_safe_departure(hazard_mean: float, hazard_sd: float, action: Action, epsilon: float = 0.05) -> float:
    """Conservative last initiation time under independent Gaussian timing uncertainty."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    if action.reliability < 1.0 - epsilon:
        return float("-inf")
    z = ND.inv_cdf(1.0 - epsilon)
    sd = math.sqrt(max(hazard_sd, 0.0) ** 2 + max(action.travel_sd, 0.0) ** 2)
    return hazard_mean - action.travel_mean - z * sd


def intervention_horizon(hazard_mean: float, hazard_sd: float, actions: Sequence[Action], epsilon: float = 0.05) -> float:
    vals = [action_last_safe_departure(hazard_mean, hazard_sd, a, epsilon) for a in actions]
    return max(vals) if vals else float("-inf")


def opportunity_volume(hazard_mean: float, hazard_sd: float, actions: Sequence[Action], epsilon: float = 0.05) -> float:
    v = viable_actions(hazard_mean, hazard_sd, actions, epsilon)
    return sum(a.reliability / (1.0 + max(a.cost, 0.0)) for a in v)


def intervention_fragility(hazard_mean: float, hazard_sd: float, actions: Sequence[Action], epsilon: float = 0.05,
                            perturb_sd: float = 3.0, n: int = 1000, seed: int = 0) -> float:
    """Monte Carlo probability that small timing perturbations erase all viable actions."""
    import numpy as np
    rng = np.random.default_rng(seed)
    lost = 0
    for _ in range(n):
        hm = hazard_mean + float(rng.normal(0, perturb_sd))
        perturbed = [Action(a.name, max(0.1, a.travel_mean + float(rng.normal(0, perturb_sd))),
                            a.travel_sd, a.reliability, a.capacity, a.cost) for a in actions]
        if not viable_actions(hm, hazard_sd, perturbed, epsilon):
            lost += 1
    return lost / n


def collective_horizon_from_individual(individual_horizons: Sequence[float], shared_delay: float = 0.0) -> float:
    """All-person collective horizon under an explicit nonnegative shared-resource delay.

    If collective success requires every person to remain preventable, joint feasibility cannot
    persist beyond the earliest individual loss of viability. A shared bottleneck can only move
    the collective deadline earlier.
    """
    if len(individual_horizons) == 0:
        return float("-inf")
    if shared_delay < 0:
        raise ValueError("shared_delay must be nonnegative")
    return min(individual_horizons) - shared_delay


def time_costed_voi(instantaneous_voi: float, delay_loss: float, observation_cost: float = 0.0) -> float:
    """Exact decomposition TCVoI = instantaneous VoI - delay loss - acquisition cost."""
    return instantaneous_voi - delay_loss - observation_cost


def status_from_horizon(horizon: float, act_threshold: float = 15.0, critical_threshold: float = 5.0) -> str:
    if not math.isfinite(horizon) or horizon <= 0:
        return "INTERVENTION_LOST"
    if horizon <= critical_threshold:
        return "CRITICAL"
    if horizon <= act_threshold:
        return "ACT"
    return "SAFE"


def analyze(hazard_mean: float, hazard_sd: float, actions: Sequence[Action], epsilon: float = 0.05,
            fragility_samples: int = 500, perturb_sd: float = 3.0, seed: int = 0,
            act_threshold: float = 15.0, critical_threshold: float = 5.0) -> HorizonResult:
    v = viable_actions(hazard_mean, hazard_sd, actions, epsilon)
    h = intervention_horizon(hazard_mean, hazard_sd, actions, epsilon)
    return HorizonResult(tuple(a.name for a in v), h, opportunity_volume(hazard_mean, hazard_sd, actions, epsilon),
                         intervention_fragility(hazard_mean, hazard_sd, actions, epsilon, perturb_sd=perturb_sd,
                                                n=fragility_samples, seed=seed),
                         status_from_horizon(h, act_threshold=act_threshold, critical_threshold=critical_threshold))
