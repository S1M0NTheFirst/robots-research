#!/usr/bin/env python3
"""
Extract and clean experiment results for academic figure generation.

Produces:
  MASTER_RESULTS.csv         — one row per run, all metrics
  fig_52_task_completion.csv — task completion by scenario/llm/condition
  fig_53_makespan_delta.csv  — makespan delta (override − baseline) per scenario/llm
  fig_55_token_consumption.csv — token costs per run + per-call overhead
  fig_56_robot_stuck.csv     — per-robot stuck rate by condition
  fig_57_model_stratification.csv — per-model avg completion rate and tool rounds

Rules:
  - gemma-3-27b excluded (all-zero results)
  - Duplicate runs for same (scenario, condition, llm) → keep latest timestamp
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

EXPERIMENTS_DIR = Path("experiments")
EXCLUDE_MODELS = {"gemma-3-27b"}


def _ts(run_dir_name: str) -> str:
    m = re.search(r"_(\d{8}-\d{6})$", run_dir_name)
    return m.group(1) if m else "00000000-000000"


def load_all_results() -> list[dict]:
    """Return deduplicated list of raw results.json dicts (gemma-3-27b removed)."""
    best: dict[tuple, tuple[str, dict]] = {}

    for path in sorted(EXPERIMENTS_DIR.glob("*/*/runs/*/results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = data.get("metadata", {})
        llm = meta.get("llm", "")
        if llm in EXCLUDE_MODELS:
            continue

        key = (meta.get("scenario"), meta.get("override_type"), llm)
        ts = _ts(path.parent.name)
        if key not in best or ts > best[key][0]:
            best[key] = (ts, data)

    records = [data for _, data in best.values()]
    print(f"Loaded {len(records)} runs (deduplicated, gemma-3-27b excluded)")
    return records


def _safe_rate(num, denom):
    return round(num / denom, 4) if denom else 0.0


def extract_master(records: list[dict]) -> list[dict]:
    rows = []
    for data in records:
        meta = data["metadata"]
        sim = data["simulation"]
        agent = data["agent"]

        n_robots = len(meta.get("all_robot_ids", []))
        n_tasks = len(meta.get("all_task_ids", []))
        total_ticks = sim.get("total_ticks", 100)
        tasks_completed = sim.get("tasks_completed", 0)
        stuck_dict = sim.get("robot_ticks_stuck", {})
        total_stuck = sum(stuck_dict.values())
        tokens_in = agent.get("total_tokens_in", 0)
        tokens_out = agent.get("total_tokens_out", 0)
        total_calls = agent.get("total_calls", 0)

        rows.append({
            "scenario": meta["scenario"],
            "condition": meta["override_type"],
            "llm": meta["llm"],
            "n_robots": n_robots,
            "n_tasks": n_tasks,
            "tasks_completed": tasks_completed,
            "tasks_completion_rate": _safe_rate(tasks_completed, n_tasks),
            "makespan": sim.get("makespan"),
            "work_tasks_never_started": sim.get("work_tasks_never_started_count", 0),
            "agent_total_calls": total_calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "tokens_in_per_call": round(tokens_in / total_calls, 1) if total_calls else 0,
            "tokens_out_per_call": round(tokens_out / total_calls, 1) if total_calls else 0,
            "tokens_total_per_call": round((tokens_in + tokens_out) / total_calls, 1) if total_calls else 0,
            "mean_tool_rounds": agent.get("mean_tool_rounds", 0),
            "decisions_truncated": agent.get("decisions_truncated_by_tool_limit", 0),
            "total_stuck_ticks": total_stuck,
            "stuck_rate_overall": _safe_rate(total_stuck, n_robots * total_ticks),
            "mean_latency_ms": agent.get("mean_latency_ms", 0),
        })
    return rows


def write_csv(rows: list[dict], filename: str, sort_keys: list[str] | None = None):
    if not rows:
        print(f"  WARNING: no data for {filename}")
        return
    if sort_keys:
        rows = sorted(rows, key=lambda r: tuple(str(r.get(k, "")) for k in sort_keys))
    fields = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {filename}  ({len(rows)} rows)")


def main():
    records = load_all_results()
    master = extract_master(records)
    write_csv(master, "MASTER_RESULTS.csv", ["scenario", "llm", "condition"])

    idx = {(r["scenario"], r["condition"], r["llm"]): r for r in master}

    # ── Figure 5.2  Task Completion ──────────────────────────────────────────
    # Long format: one row per (scenario, llm, condition)
    fig52 = [
        {
            "scenario": r["scenario"],
            "llm": r["llm"],
            "condition": r["condition"],
            "tasks_completed": r["tasks_completed"],
            "n_tasks": r["n_tasks"],
            "tasks_completion_rate": r["tasks_completion_rate"],
        }
        for r in master
    ]
    write_csv(fig52, "fig_52_task_completion.csv", ["scenario", "llm", "condition"])

    # Wide format (also useful for grouped bar charts in Excel)
    wide52: dict[tuple, dict] = {}
    for r in fig52:
        k = (r["scenario"], r["llm"])
        if k not in wide52:
            wide52[k] = {"scenario": r["scenario"], "llm": r["llm"], "n_tasks": r["n_tasks"]}
        prefix = "baseline" if r["condition"] == "baseline" else "override"
        wide52[k][f"{prefix}_tasks_completed"] = r["tasks_completed"]
        wide52[k][f"{prefix}_completion_rate"] = r["tasks_completion_rate"]

    for row in wide52.values():
        b = row.get("baseline_completion_rate", 0) or 0
        o = row.get("override_completion_rate", 0) or 0
        row["delta_completion_rate"] = round(o - b, 4)

    write_csv(
        list(wide52.values()),
        "fig_52_task_completion_wide.csv",
        ["scenario", "llm"],
    )

    # ── Figure 5.3  Makespan Delta ───────────────────────────────────────────
    b_keys = {(r["scenario"], r["llm"]) for r in master if r["condition"] == "baseline"}
    o_keys = {(r["scenario"], r["llm"]) for r in master if r["condition"] == "structured_override"}

    fig53 = []
    for scenario, llm in sorted(b_keys & o_keys):
        b = idx.get((scenario, "baseline", llm), {})
        o = idx.get((scenario, "structured_override", llm), {})
        b_ms = b.get("makespan")
        o_ms = o.get("makespan")
        delta = None if (b_ms is None or o_ms is None) else o_ms - b_ms
        fig53.append({
            "scenario": scenario,
            "llm": llm,
            "baseline_makespan": b_ms if b_ms is not None else "N/A",
            "override_makespan": o_ms if o_ms is not None else "N/A",
            "delta_makespan": delta if delta is not None else "N/A",
            "note": "negative=override faster; N/A=not all tasks completed",
        })
    write_csv(fig53, "fig_53_makespan_delta.csv")

    # ── Figure 5.5  Token Consumption ────────────────────────────────────────
    fig55 = [
        {
            "scenario": r["scenario"],
            "llm": r["llm"],
            "condition": r["condition"],
            "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"],
            "tokens_total": r["tokens_total"],
            "agent_total_calls": r["agent_total_calls"],
            "tokens_in_per_call": r["tokens_in_per_call"],
            "tokens_out_per_call": r["tokens_out_per_call"],
            "tokens_total_per_call": r["tokens_total_per_call"],
        }
        for r in master
    ]
    write_csv(fig55, "fig_55_token_consumption.csv", ["llm", "scenario", "condition"])

    # Per-model summary: avg tokens-in-per-call by condition (headline metric)
    token_summary: dict[str, dict] = {}
    for r in master:
        llm = r["llm"]
        if llm not in token_summary:
            token_summary[llm] = {"llm": llm, "b_tip": [], "o_tip": [], "b_tot": [], "o_tot": []}
        cond = r["condition"]
        if cond == "baseline":
            token_summary[llm]["b_tip"].append(r["tokens_in_per_call"])
            token_summary[llm]["b_tot"].append(r["tokens_total_per_call"])
        else:
            token_summary[llm]["o_tip"].append(r["tokens_in_per_call"])
            token_summary[llm]["o_tot"].append(r["tokens_total_per_call"])

    token_summary_rows = []
    for llm, d in sorted(token_summary.items()):
        b_tip = round(mean(d["b_tip"]), 1) if d["b_tip"] else None
        o_tip = round(mean(d["o_tip"]), 1) if d["o_tip"] else None
        b_tot = round(mean(d["b_tot"]), 1) if d["b_tot"] else None
        o_tot = round(mean(d["o_tot"]), 1) if d["o_tot"] else None
        delta_tip = round(o_tip - b_tip, 1) if (b_tip and o_tip) else None
        overhead_pct = round((o_tip - b_tip) / b_tip * 100, 1) if (b_tip and o_tip and b_tip > 0) else None
        token_summary_rows.append({
            "llm": llm,
            "avg_tokens_in_per_call_baseline": b_tip,
            "avg_tokens_in_per_call_override": o_tip,
            "delta_tokens_in_per_call": delta_tip,
            "overhead_pct": overhead_pct,
            "avg_tokens_total_per_call_baseline": b_tot,
            "avg_tokens_total_per_call_override": o_tot,
        })
    write_csv(token_summary_rows, "fig_55_token_summary_per_model.csv")

    # ── Figure 5.6  Robot Stuck Rate ─────────────────────────────────────────
    fig56 = []
    for data in records:
        meta = data["metadata"]
        sim = data["simulation"]
        scenario = meta["scenario"]
        condition = meta["override_type"]
        llm = meta["llm"]
        total_ticks = sim.get("total_ticks", 100)
        all_robots = meta.get("all_robot_ids", [])
        stuck_dict = {str(k): v for k, v in sim.get("robot_ticks_stuck", {}).items()}
        for robot_id in all_robots:
            stuck = stuck_dict.get(str(robot_id), 0)
            fig56.append({
                "scenario": scenario,
                "llm": llm,
                "condition": condition,
                "robot_id": robot_id,
                "stuck_ticks": stuck,
                "total_ticks": total_ticks,
                "stuck_rate": round(stuck / total_ticks, 4),
            })
    write_csv(fig56, "fig_56_robot_stuck.csv", ["scenario", "robot_id", "condition", "llm"])

    # Aggregated: avg stuck rate per (scenario, robot_id, condition) across all LLMs
    # Useful for a clean figure without model-level noise
    agg_key: dict[tuple, list] = defaultdict(list)
    for row in fig56:
        k = (row["scenario"], row["robot_id"], row["condition"])
        agg_key[k].append(row["stuck_rate"])

    stuck_agg = []
    for (scenario, robot_id, condition), rates in sorted(agg_key.items()):
        stuck_agg.append({
            "scenario": scenario,
            "robot_id": robot_id,
            "condition": condition,
            "avg_stuck_rate": round(mean(rates), 4),
            "n_models": len(rates),
        })
    write_csv(stuck_agg, "fig_56_robot_stuck_aggregated.csv")

    # Further aggregate to just (robot_id, condition) across ALL scenarios and models
    global_key: dict[tuple, list] = defaultdict(list)
    for row in fig56:
        k = (row["robot_id"], row["condition"])
        global_key[k].append(row["stuck_rate"])

    stuck_global = []
    for (robot_id, condition), rates in sorted(global_key.items()):
        stuck_global.append({
            "robot_id": robot_id,
            "condition": condition,
            "avg_stuck_rate": round(mean(rates), 4),
            "n_observations": len(rates),
        })
    write_csv(stuck_global, "fig_56_robot_stuck_by_robot.csv")

    # ── Figure 5.7  Model Capability Stratification ──────────────────────────
    llm_acc: dict[str, dict] = defaultdict(
        lambda: {"b_rates": [], "o_rates": [], "b_tools": [], "o_tools": []}
    )
    for r in master:
        llm = r["llm"]
        if r["condition"] == "baseline":
            llm_acc[llm]["b_rates"].append(r["tasks_completion_rate"])
            llm_acc[llm]["b_tools"].append(r["mean_tool_rounds"])
        else:
            llm_acc[llm]["o_rates"].append(r["tasks_completion_rate"])
            llm_acc[llm]["o_tools"].append(r["mean_tool_rounds"])

    fig57 = []
    for llm, d in sorted(llm_acc.items()):
        b_rate = round(mean(d["b_rates"]), 4) if d["b_rates"] else None
        o_rate = round(mean(d["o_rates"]), 4) if d["o_rates"] else None
        b_tools = round(mean(d["b_tools"]), 4) if d["b_tools"] else None
        o_tools = round(mean(d["o_tools"]), 4) if d["o_tools"] else None
        delta_rate = round(o_rate - b_rate, 4) if (b_rate is not None and o_rate is not None) else None
        delta_tools = round(o_tools - b_tools, 4) if (b_tools is not None and o_tools is not None) else None
        fig57.append({
            "llm": llm,
            "n_scenarios_baseline": len(d["b_rates"]),
            "n_scenarios_override": len(d["o_rates"]),
            "avg_completion_rate_baseline": b_rate,
            "avg_completion_rate_override": o_rate,
            "delta_completion_rate": delta_rate,
            "avg_tool_rounds_baseline": b_tools,
            "avg_tool_rounds_override": o_tools,
            "delta_tool_rounds": delta_tools,
        })
    write_csv(fig57, "fig_57_model_stratification.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
