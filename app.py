"""Interactive Intervention Horizon Research Explorer (Gradio).

Run: python app.py
Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from statistics import NormalDist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

from intervention_horizon import (
    Action,
    analyze,
    action_last_safe_departure,
    action_viability_probability,
    collective_horizon_from_individual,
    opportunity_volume,
    status_from_horizon,
    time_costed_voi,
)

ND = NormalDist()


def _actions(a_mean, a_sd, a_rel, a_cost, b_mean, b_sd, b_rel, b_cost):
    return [
        Action("Action A", a_mean, a_sd, a_rel, 20, a_cost),
        Action("Action B", b_mean, b_sd, b_rel, 20, b_cost),
    ]


def _epsilon_sweep(hazard_mean, hazard_sd, actions):
    eps = np.array([0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    rows = []
    for e in eps:
        h = max(action_last_safe_departure(hazard_mean, hazard_sd, a, float(e)) for a in actions)
        rows.append({
            "epsilon": float(e),
            "safety_confidence": 1.0 - float(e),
            "horizon_min": h if np.isfinite(h) else np.nan,
            "opportunity_volume": opportunity_volume(hazard_mean, hazard_sd, actions, float(e)),
            "status": status_from_horizon(h),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(df["epsilon"], df["horizon_min"], marker="o")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Risk tolerance epsilon")
    ax.set_ylabel("Intervention Horizon (min)")
    ax.set_title("Single-case sensitivity to required safety confidence")
    fig.tight_layout()
    return df, fig


def evaluate(
    hazard_mean, hazard_sd, epsilon, perturb, act_threshold, critical_threshold,
    a_mean, a_sd, a_rel, a_cost, b_mean, b_sd, b_rel, b_cost,
    voi, delay_loss, obs_cost, people, earliest, spread, shared_delay,
):
    actions = _actions(a_mean, a_sd, a_rel, a_cost, b_mean, b_sd, b_rel, b_cost)
    result = analyze(
        hazard_mean, hazard_sd, actions, epsilon,
        fragility_samples=1500, perturb_sd=perturb, seed=20260827,
        act_threshold=act_threshold, critical_threshold=critical_threshold,
    )

    rows = []
    for a in actions:
        p = action_viability_probability(hazard_mean, hazard_sd, a)
        d = action_last_safe_departure(hazard_mean, hazard_sd, a, epsilon)
        rows.append({
            "action": a.name,
            "travel_mean_min": a.travel_mean,
            "travel_sd_min": a.travel_sd,
            "reliability": a.reliability,
            "normalized_cost": a.cost,
            "success_probability": p,
            "required_probability": 1.0 - epsilon,
            "last_safe_departure_min": d if np.isfinite(d) else np.nan,
            "viable_now": a.name in result.viable_actions,
        })
    action_df = pd.DataFrame(rows)

    tc = time_costed_voi(voi, delay_loss, obs_cost)
    info_df = pd.DataFrame([
        {"component": "Instantaneous VoI", "value": voi},
        {"component": "Delay loss", "value": -delay_loss},
        {"component": "Observation cost", "value": -obs_cost},
        {"component": "Time-costed VoI", "value": tc},
    ])

    n = int(people)
    individual = np.linspace(earliest, earliest + spread, n)
    coll = collective_horizon_from_individual(individual.tolist(), shared_delay)
    collective_df = pd.DataFrame({
        "quantity": ["People", "Earliest individual horizon", "Latest individual horizon", "Shared-resource delay", "Collective horizon", "Bound holds"],
        "value": [n, float(individual.min()), float(individual.max()), shared_delay, coll, bool(coll <= individual.min() + 1e-12)],
    })

    sens_df, sens_fig = _epsilon_sweep(hazard_mean, hazard_sd, actions)

    summary = (
        f"### Decision state: **{result.status}**\n"
        f"- Intervention Horizon: **{result.horizon:.2f} min**\n"
        f"- Viable interventions: **{', '.join(result.viable_actions) if result.viable_actions else 'none'}**\n"
        f"- Opportunity volume: **{result.opportunity_volume:.3f}**\n"
        f"- Intervention fragility: **{result.fragility:.3f}** under perturbation SD {perturb:.1f} min\n"
        f"- Required safety confidence: **{1-epsilon:.3f}**\n"
        f"- Time-costed VoI: **{tc:.3f}** -> **{'OBSERVE' if tc > 0 else 'ACT NOW'}**\n"
        f"- Collective horizon: **{coll:.2f} min** (earliest individual: {individual.min():.2f} min)\n"
        f"- Collective theorem bound verified: **{coll <= individual.min() + 1e-12}**"
    )
    return summary, action_df, sens_df, sens_fig, info_df, collective_df


with gr.Blocks(title="Intervention Horizon Research Explorer") as demo:
    gr.Markdown(
        "# Intervention Horizon Research Explorer\n"
        "Change hazard, uncertainty, safety confidence, access, reliability, cost, information delay, "
        "decision thresholds, and shared-resource parameters. Outputs cover single-case viability, "
        "horizon, opportunity volume, fragility, epsilon sensitivity, time-costed value of information, "
        "and collective feasibility. This is a research reference implementation, not an operational warning service."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Hazard and decision policy")
            hazard_mean = gr.Slider(1, 180, 35, step=.5, label="Expected hazard lead (min)")
            hazard_sd = gr.Slider(0, 30, 4, step=.5, label="Hazard timing SD (min)")
            epsilon = gr.Slider(.005, .30, .05, step=.005, label="Risk tolerance epsilon")
            perturb = gr.Slider(0, 15, 3, step=.5, label="Fragility perturbation SD (min)")
            act_threshold = gr.Slider(1, 45, 15, step=.5, label="ACT threshold (min)")
            critical_threshold = gr.Slider(0, 20, 5, step=.5, label="CRITICAL threshold (min)")
            gr.Markdown("### Action A")
            a_mean = gr.Slider(.5, 90, 8, step=.5, label="Travel time")
            a_sd = gr.Slider(0, 20, 1.5, step=.5, label="Travel-time SD")
            a_rel = gr.Slider(.5, 1, .995, step=.001, label="Reliability")
            a_cost = gr.Slider(0, 10, .2, step=.1, label="Normalized cost")
        with gr.Column():
            gr.Markdown("### Action B")
            b_mean = gr.Slider(.5, 90, 22, step=.5, label="Travel time")
            b_sd = gr.Slider(0, 20, 2.5, step=.5, label="Travel-time SD")
            b_rel = gr.Slider(.5, 1, .995, step=.001, label="Reliability")
            b_cost = gr.Slider(0, 10, .5, step=.1, label="Normalized cost")
            gr.Markdown("### Observe or act")
            voi = gr.Slider(0, 1, .15, step=.01, label="Instantaneous VoI")
            delay_loss = gr.Slider(0, 1, .10, step=.01, label="Delay-induced opportunity loss")
            obs_cost = gr.Slider(0, .5, .01, step=.01, label="Observation cost")
            gr.Markdown("### Collective horizon")
            people = gr.Slider(2, 250, 20, step=1, label="People")
            earliest = gr.Slider(0, 90, 15, step=.5, label="Earliest individual horizon")
            spread = gr.Slider(0, 60, 10, step=.5, label="Individual-horizon spread")
            shared_delay = gr.Slider(0, 60, 3, step=.5, label="Shared-resource delay")

    run = gr.Button("Evaluate all outputs", variant="primary")
    summary = gr.Markdown()
    with gr.Tab("Action viability"):
        action_table = gr.Dataframe(label="Action-level viability and deadlines")
    with gr.Tab("Safety-confidence sensitivity"):
        sensitivity_table = gr.Dataframe(label="Epsilon sensitivity")
        sensitivity_plot = gr.Plot(label="Horizon sensitivity")
    with gr.Tab("Observe or act"):
        info_table = gr.Dataframe(label="TCVoI decomposition")
    with gr.Tab("Collective feasibility"):
        collective_table = gr.Dataframe(label="Collective-horizon diagnostics")

    inputs = [
        hazard_mean, hazard_sd, epsilon, perturb, act_threshold, critical_threshold,
        a_mean, a_sd, a_rel, a_cost, b_mean, b_sd, b_rel, b_cost,
        voi, delay_loss, obs_cost, people, earliest, spread, shared_delay,
    ]
    outputs = [summary, action_table, sensitivity_table, sensitivity_plot, info_table, collective_table]
    run.click(evaluate, inputs=inputs, outputs=outputs)
    demo.load(evaluate, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
