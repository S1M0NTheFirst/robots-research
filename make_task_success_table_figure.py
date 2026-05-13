"""
Render the Task Success Rate table as a publication-quality figure,
matching the LaTeX table format: green cells where SAI > Base,
red cells where SAI < Base, bold for the winning value.

Produces:
  figures/fig_task_success_table.png
"""

import os
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from decimal import Decimal, ROUND_HALF_UP

os.makedirs('figures', exist_ok=True)

LLM_ORDER = [
    'asi1', 'claude-haiku', 'deepseek-r1', 'gemini-2.0-flash',
    'gemma-4-26b-a4b', 'gpt-4o', 'gpt-oss-20b', 'llama-3.3-70b', 'qwen3-32b',
]
LLM_LABELS = [
    'ASI1', 'Claude-Haiku', 'DeepSeek-R1', 'Gemini-2.0-Flash',
    'Gemma-4-26B', 'GPT-4o', 'GPT-OSS-20B', 'LLaMA-3.3-70B', 'Qwen3-32B',
]
SCENARIOS = list(range(1, 9))


def round2(x):
    return float(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


# ── Extract data ──────────────────────────────────────────────────────────────
def load_rates():
    base_grid = {}
    sai_grid  = {}
    for s in SCENARIOS:
        wb = openpyxl.load_workbook(f'scenario_{s:02d}.xlsx')
        task_sets = {}
        for row in list(wb['tasks'].iter_rows(values_only=True))[1:]:
            _, override_type, llm, _, task_id = row[:5]
            if llm not in LLM_ORDER:
                continue
            task_sets.setdefault((override_type, llm), set()).add(task_id)
        run_best = {}
        for row in list(wb['summary'].iter_rows(values_only=True))[1:]:
            _, override_type, llm, _, _, _, tasks_completed = row[:7]
            if llm not in LLM_ORDER:
                continue
            key = (override_type, llm)
            run_best[key] = max(run_best.get(key, 0), tasks_completed or 0)
        for llm in LLM_ORDER:
            for override, grid in [('baseline', base_grid), ('structured_override', sai_grid)]:
                key = (override, llm)
                total = len(task_sets.get(key, set()))
                completed = run_best.get(key, 0)
                grid[(llm, s)] = round2(completed / total) if total > 0 else None

    return base_grid, sai_grid


base_grid, sai_grid = load_rates()

# ── Build cell data ───────────────────────────────────────────────────────────
# Each data cell: (text, bg_color, bold)
GREEN = '#c8e6c9'   # light green
RED   = '#ffcdd2'   # light red
WHITE = 'white'

n_llm = len(LLM_LABELS)
n_scen = len(SCENARIOS)

# cells[row][col] = (text, bg, bold)
# Columns: LLM label + 2 columns per scenario (Base, SAI)
n_cols = 1 + n_scen * 2

cells = []
for i, llm in enumerate(LLM_ORDER):
    row = []
    row.append((LLM_LABELS[i], WHITE, False))   # LLM name cell
    for s in SCENARIOS:
        base = base_grid.get((llm, s))
        sai  = sai_grid.get((llm, s))

        # Format text
        base_txt = f'{base:.2f}' if base is not None else '–'
        sai_txt  = f'{sai:.2f}'  if sai  is not None else '–'

        # Determine colors and bold
        if base is None or sai is None:
            base_bg, sai_bg = WHITE, WHITE
            base_bold = base is not None
            sai_bold  = sai  is not None
        elif sai > base:
            base_bg, sai_bg = WHITE, GREEN
            base_bold, sai_bold = False, True
        elif sai < base:
            base_bg, sai_bg = WHITE, RED
            base_bold, sai_bold = True, False
        else:
            base_bg, sai_bg = WHITE, WHITE
            base_bold, sai_bold = False, False

        row.append((base_txt, base_bg, base_bold))
        row.append((sai_txt,  sai_bg,  sai_bold))
    cells.append(row)


# ── Draw ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'Arial'})

fig_w = 18
row_h = 0.42
header_h = 0.72   # two header rows combined
fig_h = header_h + n_llm * row_h + 0.3

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')
ax.set_axis_off()

# Column widths (relative, sum = 1)
llm_w = 0.13
col_w = (1.0 - llm_w) / (n_scen * 2)

# Compute x positions for each column
x_positions = [0]   # left edge of each column
x_positions.append(llm_w)
for _ in range(n_scen * 2):
    x_positions.append(x_positions[-1] + col_w)

total_rows = n_llm + 3   # 2 header rows + 1 subheader + data rows
row_heights = [header_h / 2] * 2 + [row_h] * (n_llm + 1)  # +1 for subheader

# y positions (top to bottom)
y_tops = [1.0]
for h in row_heights:
    y_tops.append(y_tops[-1] - h / fig_h)

def draw_cell(ax, x, y, w, h, text, bg, bold, fontsize=9, ha='center', va='center'):
    rect = mpatches.FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle='square,pad=0',
        facecolor=bg,
        edgecolor='#cccccc',
        linewidth=0.4,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + w / 2, y - h / 2, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize,
        fontweight=weight,
        fontfamily='Arial',
        color='#111111',
    )

# ── Row 0: title header "S1 … S8" ────────────────────────────────────────────
hr0 = row_heights[0] / fig_h
y0  = y_tops[0]

# LLM column header (spans 2 header rows)
draw_cell(ax, x_positions[0], y0, llm_w, 2 * hr0, 'LLM',
          'white', True, fontsize=10, va='center')

for si, s in enumerate(SCENARIOS):
    col_start = x_positions[1 + si * 2]
    group_w   = col_w * 2
    draw_cell(ax, col_start, y0, group_w, hr0,
              f'S{s}', '#f0f0f0', True, fontsize=9)

# ── Row 1: "Base / SAI" sub-headers ──────────────────────────────────────────
hr1 = row_heights[1] / fig_h
y1  = y_tops[1]

for si in range(n_scen):
    for j, lbl in enumerate(['Base', 'SAI']):
        cx = x_positions[1 + si * 2 + j]
        draw_cell(ax, cx, y1, col_w, hr1, lbl, '#f0f0f0', False, fontsize=8.5)

# ── Separator line under headers ──────────────────────────────────────────────
sep_y = y_tops[2]
ax.plot([0, 1], [sep_y, sep_y], color='#333333', linewidth=1.2,
        transform=ax.transAxes, clip_on=False)

# ── Data rows ─────────────────────────────────────────────────────────────────
for ri, row in enumerate(cells):
    row_idx = ri + 2   # offset for 2 header rows
    rh = row_heights[row_idx] / fig_h
    yt = y_tops[row_idx]
    bg_row = '#fafafa' if ri % 2 == 1 else 'white'

    for ci, (text, bg, bold) in enumerate(row):
        cx = x_positions[ci]
        cw = llm_w if ci == 0 else col_w
        cell_bg = bg if bg != 'white' else bg_row
        cell_ha = 'left' if ci == 0 else 'center'
        px = cx + 0.005 if ci == 0 else cx
        draw_cell(ax, px if ci == 0 else cx, yt, cw, rh,
                  text, cell_bg, bold,
                  fontsize=9 if ci == 0 else 8.5,
                  ha=cell_ha)

# Bottom border
ax.plot([0, 1], [y_tops[2 + n_llm], y_tops[2 + n_llm]], color='#333333',
        linewidth=1.2, transform=ax.transAxes, clip_on=False)

# Top border
ax.plot([0, 1], [y_tops[0], y_tops[0]], color='#333333',
        linewidth=1.2, transform=ax.transAxes, clip_on=False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=GREEN, edgecolor='#aaaaaa', label='SAI > Baseline'),
    mpatches.Patch(facecolor=RED,   edgecolor='#aaaaaa', label='SAI < Baseline'),
    mpatches.Patch(facecolor='white', edgecolor='#aaaaaa', label='Tied'),
]
ax.legend(handles=legend_elements, loc='lower right',
          bbox_to_anchor=(1.0, -0.08),
          frameon=True, fontsize=9, ncol=3,
          handlelength=1.2, handleheight=0.9)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Task Success Rate: Baseline vs. SAI across 8 Scenarios',
             fontsize=12, fontweight='bold', pad=8, fontfamily='Arial',
             transform=ax.transAxes, x=0.5, y=y_tops[0] + 0.04)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

out = 'figures/fig_task_success_table.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
