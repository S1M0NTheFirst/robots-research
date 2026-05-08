"""
Normalized stacked bar chart — Robot time allocation per scenario.

For each scenario we show two bars (Baseline / SAI), averaged across all
LLMs and all robots in that run.  The four segments always sum to 100 %:

  Working   — robot was actively doing work on a task
  Moving    — robot was travelling toward its assigned task
  Stuck     — robot tried to move but was blocked / no valid path
              (carved out of the idle bucket; pure_idle = idle − stuck ≥ 0)
  Pure Idle — robot had nothing to do (no assignment, or task already done)

Saved to: figures/fig_robot_timeshare.png
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openpyxl

os.makedirs('figures', exist_ok=True)

# ── 1. Load robots data from all scenario xlsx files ─────────────────────────
all_rows = []
for i in range(1, 9):
    wb = openpyxl.load_workbook(f'scenario_{i:02d}.xlsx')
    ws = wb['robots']
    headers = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if any(v is not None for v in row):
            d = dict(zip(headers, row))
            all_rows.append(d)

# Exclude gemma-3-27b (consistently failed; excluded from all analyses)
all_rows = [r for r in all_rows if r['llm'] != 'gemma-3-27b']

# ── 2. Aggregate: mean per (scenario, condition) ───────────────────────────
agg = defaultdict(lambda: {'w': [], 'm': [], 'idle': [], 'stuck': []})
for r in all_rows:
    key = (r['scenario'], r['override_type'])
    agg[key]['w'].append(r['ticks_working'] or 0)
    agg[key]['m'].append(r['ticks_moving']  or 0)
    agg[key]['idle'].append(r['ticks_idle'] or 0)
    agg[key]['stuck'].append(r['ticks_stuck'] or 0)

SCENARIOS   = [f'scenario_{i:02d}' for i in range(1, 9)]
CONDITIONS  = ['baseline', 'structured_override']
COND_LABELS = {'baseline': 'Baseline', 'structured_override': 'SAI'}

segments = {}   # (scenario, cond) → [working, moving, stuck, pure_idle]
for sc in SCENARIOS:
    for cond in CONDITIONS:
        key = (sc, cond)
        if key not in agg:
            segments[key] = [0, 0, 0, 0]
            continue
        vals = agg[key]
        n       = len(vals['w'])
        avg_w   = sum(vals['w'])    / n
        avg_m   = sum(vals['m'])    / n
        avg_i   = sum(vals['idle']) / n
        avg_s   = sum(vals['stuck'])/ n
        pure_i  = max(0.0, avg_i - avg_s)
        segments[key] = [avg_w, avg_m, avg_s, pure_i]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})

COLORS = {
    'Working':   '#2d8f4e',   # dark green
    'Moving':    '#4a90d9',   # steel blue
    'Stuck':     '#d9534f',   # red
    'Pure Idle': '#d0d0d0',   # light gray
}
SEG_KEYS   = ['Working', 'Moving', 'Stuck', 'Pure Idle']

n_scenarios = len(SCENARIOS)
n_conds     = len(CONDITIONS)
bar_w       = 0.35
group_gap   = 0.15                        # gap between conditions inside one group
group_w     = n_conds * bar_w + group_gap # total width of one scenario group

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x_ticks = []
x_labels = []

for g_idx, sc in enumerate(SCENARIOS):
    group_centre = g_idx * (group_w + 0.3)
    for c_idx, cond in enumerate(CONDITIONS):
        x = group_centre + c_idx * (bar_w + group_gap / n_conds)
        segs = segments[(sc, cond)]        # [working, moving, stuck, pure_idle]
        bottom = 0.0
        for s_idx, (seg_name, val) in enumerate(zip(SEG_KEYS, segs)):
            color = COLORS[seg_name]
            # Slight transparency for SAI bars so baseline is more prominent
            alpha = 1.0 if cond == 'baseline' else 0.78
            ax.bar(x, val, bar_w, bottom=bottom, color=color, alpha=alpha,
                   edgecolor='white', linewidth=0.4)
            bottom += val

        # Label each bar with condition abbreviation
        label = 'Base' if cond == 'baseline' else 'SAI'
        ax.text(x, -3.5, label, ha='center', va='top', fontsize=9,
                fontfamily='Arial', color='#444444')

    # Scenario tick mark at group centre
    x_ticks.append(group_centre + (bar_w + group_gap / n_conds) / 2)
    x_labels.append(f'S{g_idx+1}')

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=13, fontfamily='Arial')
ax.set_xlim(-0.4, x_ticks[-1] + group_w)
ax.set_ylim(-8, 108)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f'{v}%' for v in range(0, 101, 20)], fontsize=12, fontfamily='Arial')
ax.set_ylabel('% of Simulation Ticks', fontsize=14, labelpad=8, fontfamily='Arial')
ax.set_xlabel('Scenario', fontsize=14, labelpad=22, fontfamily='Arial')
ax.set_title('Robot Time Allocation per Scenario\n(averaged across all LLMs and robots)',
             fontsize=15, fontweight='bold', pad=12, fontfamily='Arial')

# Gridlines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, color='#aaaaaa')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend
patches = [mpatches.Patch(color=COLORS[k], label=k) for k in SEG_KEYS]
ax.legend(handles=patches, loc='upper right', fontsize=11,
          framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
out = 'figures/fig_robot_timeshare.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
