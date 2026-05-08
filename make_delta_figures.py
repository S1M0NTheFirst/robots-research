"""
Generate delta heatmaps (SAI - Baseline) showing the difference between conditions.
Green = SAI improved over Baseline, Red = SAI worse than Baseline, White = no change.

Produces:
  figures/fig_completion_delta.png  -- task completion rate delta
  figures/fig_tokens_delta.png      -- avg tokens/call delta
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('figures', exist_ok=True)

# ── LLM ordering: best baseline completion rate first ────────────────────────
LLM_ORDER = [
    'gemma-4-26b-a4b',
    'gpt-4o',
    'claude-haiku',
    'deepseek-r1',
    'asi1',
    'gemini-2.0-flash',
    'llama-3.3-70b',
    'qwen3-32b',
    'gpt-oss-20b',
]
LLM_LABELS = [
    'Gemma-4-26B',
    'GPT-4o',
    'Claude-Haiku',
    'DeepSeek-R1',
    'ASI1',
    'Gemini-2.0-Flash',
    'LLaMA-3.3-70B',
    'Qwen3-32B',
    'GPT-OSS-20B',
]

SCENARIOS = list(range(1, 9))


# ── Build pivot grid (LLMs x Scenarios) ──────────────────────────────────────
def load_grid(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    sub = df[df['condition'] == condition][['llm', 'scenario', metric]].copy()
    sub['scen_num'] = sub['scenario'].str.extract(r'(\d+)').astype(int)
    pivot = sub.pivot_table(index='llm', columns='scen_num', values=metric)
    pivot = pivot.reindex(index=LLM_ORDER, columns=SCENARIOS)
    pivot.index = LLM_LABELS
    pivot.columns = [f'S{s}' for s in SCENARIOS]
    return pivot


# ── Draw a single delta heatmap ───────────────────────────────────────────────
def make_delta_heatmap(
    delta: pd.DataFrame,
    title: str,
    cbar_label: str,
    fmt: str,
    out_path: str,
):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})

    # Symmetric colour scale centred on 0
    abs_max = np.nanmax(np.abs(delta.values))
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')

    annot = delta.map(lambda v: fmt % v if not np.isnan(v) else 'N/A')

    sns.heatmap(
        delta,
        ax=ax,
        annot=annot,
        fmt='',
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        center=0,
        linewidths=0.5,
        linecolor='#dddddd',
        cbar=True,
        cbar_kws={'label': cbar_label, 'shrink': 0.85},
        annot_kws={'size': 11, 'family': 'Arial'},
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=12, fontfamily='Arial')
    ax.set_xlabel('Scenario', fontsize=16, labelpad=8, fontfamily='Arial')
    ax.set_ylabel('LLM', fontsize=16, labelpad=8, fontfamily='Arial')
    ax.tick_params(axis='x', labelsize=13, rotation=0)
    ax.tick_params(axis='y', labelsize=13, rotation=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('Arial')
    ax.figure.axes[-1].yaxis.label.set_fontfamily('Arial')
    ax.figure.axes[-1].yaxis.label.set_fontsize(14)
    ax.figure.axes[-1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('MASTER_RESULTS.csv')

# ── Completion rate delta (positive = SAI improved) ───────────────────────────
cmp_base  = load_grid(df, 'baseline',            'tasks_completion_rate')
cmp_sai   = load_grid(df, 'structured_override', 'tasks_completion_rate')
cmp_delta = cmp_sai - cmp_base

make_delta_heatmap(
    cmp_delta,
    title='Task Completion Rate: SAI - Baseline',
    cbar_label='Delta Completion Rate',
    fmt='%+.2f',
    out_path='figures/fig_completion_delta.png',
)

# ── Token delta (positive = SAI uses more tokens) ─────────────────────────────
tok_base  = load_grid(df, 'baseline',            'tokens_in_per_call')
tok_sai   = load_grid(df, 'structured_override', 'tokens_in_per_call')
tok_delta = tok_sai - tok_base

make_delta_heatmap(
    tok_delta,
    title='Avg Tokens / Call: SAI - Baseline',
    cbar_label='Delta Tokens / Call',
    fmt='%+.0f',
    out_path='figures/fig_tokens_delta.png',
)

print('\nDone. Check figures/')
