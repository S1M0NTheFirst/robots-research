"""
Generate individual annotated heatmap figures (4 total):
  figures/fig_tokens_baseline.png
  figures/fig_tokens_sai.png
  figures/fig_completion_baseline.png
  figures/fig_completion_sai.png
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


# ── Draw a single heatmap ─────────────────────────────────────────────────────
def make_single_heatmap(
    data: pd.DataFrame,
    title: str,
    cbar_label: str,
    fmt: str,
    out_path: str,
    cmap: str,
    vmin: float,
    vmax: float,
):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 16})

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')

    annot = data.map(lambda v: fmt % v if not np.isnan(v) else 'N/A')

    sns.heatmap(
        data,
        ax=ax,
        annot=annot,
        fmt='',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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


# ── Load master data ──────────────────────────────────────────────────────────
df = pd.read_csv('MASTER_RESULTS.csv')

# ── Tokens per call ───────────────────────────────────────────────────────────
tok_base = load_grid(df, 'baseline',            'tokens_in_per_call')
tok_sai  = load_grid(df, 'structured_override', 'tokens_in_per_call')
tok_vmin = min(tok_base.min().min(), tok_sai.min().min())
tok_vmax = max(tok_base.max().max(), tok_sai.max().max())

make_single_heatmap(tok_base, '(a) Baseline',
                    'Avg Tokens / Call', '%.0f',
                    'figures/fig_tokens_baseline.png', 'Blues',
                    tok_vmin, tok_vmax)

make_single_heatmap(tok_sai,  '(b) SAI',
                    'Avg Tokens / Call', '%.0f',
                    'figures/fig_tokens_sai.png', 'Blues',
                    tok_vmin, tok_vmax)

# ── Task completion rate ──────────────────────────────────────────────────────
cmp_base = load_grid(df, 'baseline',            'tasks_completion_rate')
cmp_sai  = load_grid(df, 'structured_override', 'tasks_completion_rate')
cmp_vmin = min(cmp_base.min().min(), cmp_sai.min().min())
cmp_vmax = max(cmp_base.max().max(), cmp_sai.max().max())

make_single_heatmap(cmp_base, '(a) Baseline',
                    'Task Completion Rate', '%.2f',
                    'figures/fig_completion_baseline.png', 'YlGn',
                    cmp_vmin, cmp_vmax)

make_single_heatmap(cmp_sai,  '(b) SAI',
                    'Task Completion Rate', '%.2f',
                    'figures/fig_completion_sai.png', 'YlGn',
                    cmp_vmin, cmp_vmax)

print('\nDone. 4 figures saved to figures/')
