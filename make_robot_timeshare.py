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
plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})

COLORS = {
    'Working':   '#2d8f4e',   # dark green
    'Moving':    '#4a90d9',   # steel blue
    'Stuck':     '#d9534f',   # red
    'Pure Idle': '#d0d0d0',   # light gray
}
SEG_KEYS   = ['Working', 'Moving', 'Stuck', 'Pure Idle']

bar_w     = 0.32
gap       = 0.08
group_gap = 0.72   # widened so 8 scenario groups span the same x-range as
                   # the 9-LLM figure → bars render at identical physical width

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x_ticks = []
x_labels = []

for g_idx, sc in enumerate(SCENARIOS):
    group_centre = g_idx * (2 * bar_w + gap + group_gap)
    for c_idx, cond in enumerate(CONDITIONS):
        x = group_centre + c_idx * (bar_w + gap)
        segs = segments[(sc, cond)]
        bottom = 0.0
        for seg_name, val in zip(SEG_KEYS, segs):
            ax.bar(x, val, bar_w, bottom=bottom, color=COLORS[seg_name],
                   alpha=1.0, edgecolor='white', linewidth=0.4)
            bottom += val

        label = 'Base' if cond == 'baseline' else 'SAI'
        ax.text(x, -3.5, label, ha='center', va='top', fontsize=13,
                fontfamily='Arial', color='#444444')

    pair_centre = group_centre + (bar_w + gap) / 2
    x_ticks.append(pair_centre)
    x_labels.append(f'S{g_idx+1}')

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=15, fontfamily='Arial')
ax.set_xlim(-0.4, x_ticks[-1] + bar_w + group_gap / 2)
ax.set_ylim(-8, 108)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f'{v}%' for v in range(0, 101, 20)], fontsize=16, fontfamily='Arial')
ax.set_ylabel('% of Simulation Ticks', fontsize=16, labelpad=8, fontfamily='Arial')
ax.set_xlabel('Scenario', fontsize=16, labelpad=12, fontfamily='Arial')

# Gridlines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, color='#aaaaaa')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend
patches = [mpatches.Patch(color=COLORS[k], label=k) for k in SEG_KEYS]
ax.legend(handles=patches, loc='upper right', fontsize=16,
          framealpha=0.9, edgecolor='#cccccc')

plt.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.22)
out = 'figures/fig_robot_timeshare.png'
plt.savefig(out, dpi=300, facecolor='white')
pdf_out = out.replace('.png', '.pdf')
plt.savefig(pdf_out, facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
print(f'Saved: {pdf_out}')
