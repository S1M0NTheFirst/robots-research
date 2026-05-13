"""
Stacked bar chart — Robot time allocation per LLM.

X-axis: 9 LLMs (ordered by baseline task completion rate, matching other figures)
Each LLM has two bars: Baseline and SAI.
Data is averaged across all 8 scenarios and all robots within each (llm, condition).

Four segments (always sum to 100%):
  Working   — actively doing work on a task
  Moving    — travelling toward a task
  Stuck     — tried to move but was blocked
  Pure Idle — no assignment / waiting  (= idle − stuck)

Saved to: figures/fig_robot_llm_timeshare.png
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

# ── 1. Load data ──────────────────────────────────────────────────────────────
all_rows = []
for i in range(1, 9):
    wb = openpyxl.load_workbook(f'scenario_{i:02d}.xlsx')
    ws = wb['robots']
    headers = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if any(v is not None for v in row):
            all_rows.append(dict(zip(headers, row)))

all_rows = [r for r in all_rows if r['llm'] != 'gemma-3-27b']

# ── 2. Aggregate per (llm, condition) across all scenarios ────────────────────
agg = defaultdict(lambda: {'w': [], 'm': [], 'idle': [], 'stuck': []})
for r in all_rows:
    key = (r['llm'], r['override_type'])
    agg[key]['w'].append(r['ticks_working'] or 0)
    agg[key]['m'].append(r['ticks_moving']  or 0)
    agg[key]['idle'].append(r['ticks_idle'] or 0)
    agg[key]['stuck'].append(r['ticks_stuck'] or 0)

LLM_ORDER = [
    'gemma-4-26b-a4b', 'gpt-4o', 'claude-haiku', 'deepseek-r1',
    'asi1', 'gemini-2.0-flash', 'llama-3.3-70b', 'qwen3-32b', 'gpt-oss-20b',
]
LLM_LABELS = [
    'Gemma-4-26B', 'GPT-4o', 'Claude-Haiku', 'DeepSeek-R1',
    'ASI1', 'Gemini-2.0-Flash', 'LLaMA-3.3-70B', 'Qwen3-32B', 'GPT-OSS-20B',
]
CONDITIONS  = ['baseline', 'structured_override']

def get_segments(llm, cond):
    key = (llm, cond)
    vals = agg[key]
    n = len(vals['w'])
    if n == 0:
        return [0, 0, 0, 0]
    avg_w  = sum(vals['w'])    / n
    avg_m  = sum(vals['m'])    / n
    avg_s  = sum(vals['stuck'])/ n
    avg_i  = sum(vals['idle']) / n
    pure_i = max(0.0, avg_i - avg_s)
    return [avg_w, avg_m, avg_s, pure_i]   # [working, moving, stuck, pure_idle]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})

COLORS = {
    'Working':   '#2d8f4e',
    'Moving':    '#4a90d9',
    'Stuck':     '#d9534f',
    'Pure Idle': '#d0d0d0',
}
SEG_KEYS = ['Working', 'Moving', 'Stuck', 'Pure Idle']

bar_w     = 0.32
gap       = 0.08          # gap between baseline and SAI within a group
group_gap = 0.55          # gap between LLM groups

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x_ticks, x_labels = [], []

for g_idx, (llm, label) in enumerate(zip(LLM_ORDER, LLM_LABELS)):
    group_centre = g_idx * (2 * bar_w + gap + group_gap)

    for c_idx, cond in enumerate(CONDITIONS):
        x = group_centre + c_idx * (bar_w + gap)
        segs = get_segments(llm, cond)
        bottom = 0.0
        for seg_name, val in zip(SEG_KEYS, segs):
            ax.bar(x, val, bar_w, bottom=bottom,
                   color=COLORS[seg_name], alpha=1.0,
                   edgecolor='white', linewidth=0.4)
            bottom += val

        cond_label = 'Base' if cond == 'baseline' else 'SAI'
        ax.text(x, -3.5, cond_label, ha='center', va='top',
                fontsize=13, fontfamily='Arial', color='#444444')

    # LLM label centred under the pair
    pair_centre = group_centre + (bar_w + gap) / 2
    x_ticks.append(pair_centre)
    x_labels.append(label)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=15, fontfamily='Arial', rotation=15, ha='right')
ax.set_xlim(-0.4, x_ticks[-1] + bar_w + group_gap / 2)
ax.set_ylim(-8, 108)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f'{v}%' for v in range(0, 101, 20)], fontsize=16, fontfamily='Arial')
ax.set_ylabel('% of Simulation Ticks', fontsize=16, labelpad=8, fontfamily='Arial')
ax.set_xlabel('LLM', fontsize=16, labelpad=12, fontfamily='Arial')

ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, color='#aaaaaa')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

patches = [mpatches.Patch(color=COLORS[k], label=k) for k in SEG_KEYS]
ax.legend(handles=patches, loc='upper right', fontsize=16,
          framealpha=0.9, edgecolor='#cccccc')

plt.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.22)
out = 'figures/fig_robot_llm_timeshare.png'
plt.savefig(out, dpi=300, facecolor='white')
pdf_out = out.replace('.png', '.pdf')
plt.savefig(pdf_out, facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
print(f'Saved: {pdf_out}')
