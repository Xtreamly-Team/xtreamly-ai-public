# -*- coding: utf-8 -*-
"""
This concept focuses on evaluating volatility predictions, particularly during high-volatility periods.
The data is right-skewed, making it logical to target outliers in the upper tail.
Resampled values resemble a Generalized Extreme Value (GEV) distribution (e.g., Weibull, Exponential).

By dividing volatility by the mean over the past 4 hours, we obtain a normalized score.
Thresholds can then be applied to both the raw predictions and these scores.
The score also helps identify local outliers, which is valuable given the seasonal nature of volatility.
"""
# %reset -f
import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import pytz
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
pd.set_option('display.max_columns', None)
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from settings.plot import tailwind, _style, _style_white

folder = 'scores_example'

# Actual volatilites
df_std = pd.read_csv(os.path.join('..', 'data', f'df_std.csv'))
df_std = df_std[['_time', 'symbol', 'price', 'std_1min']]
df_std['_time'] = pd.to_datetime(df_std['_time']).dt.tz_localize(None)
df_std['std_1min'] /= 100
data_std = []
for s in ['BTC', 'ETH']:
    df_s = df_std[df_std['symbol']==s]
    df_s.index = df_s.pop('_time')
    df_s = df_s.sort_index()
    df_s['std_1min_mean'] = df_s['std_1min'].shift(-1).rolling(window=f'240min').mean() 
    data_std += [df_s]
df_std = pd.concat(data_std).reset_index()

# Xtreamly predicted states
df_state_xtreamly = pd.read_csv(os.path.join('..', 'data', f'df_state_xtreamly.csv'))
df_state = df_state_xtreamly[['_time', 'symbol', '1min']]
df_state['_time'] = pd.to_datetime(df_state['_time']).dt.tz_localize(None)
df_state = df_state.rename(columns={'1min': 'pred_1min'})
df_state = df_state.merge(df_std, on=['_time', 'symbol'], how='left')
df_state['score_1min'] = df_state['pred_1min']/df_state['std_1min_mean']

# =============================================================================
# Threshold
# =============================================================================
thres_pred= df_state['pred_1min'].quantile(.99)
thres_score = df_state['score_1min'].quantile(.99)

df_state['extreme_highvol'] = (df_state['pred_1min'] >= thres_pred) | (df_state['score_1min'] >= thres_score)
# =============================================================================
# Color
# =============================================================================
palette_pred = [
    tailwind['blue-500'],
    tailwind['teal-500'],
    tailwind['amber-500'],
    tailwind['yellow-500'],
    tailwind['yellow-700'],
    tailwind['orange-700'],
    ]
palette_score = [
    tailwind['cyan-100'],
    tailwind['cyan-900'],
    ]
def _pred_color(pred, palette):
    cmap = LinearSegmentedColormap.from_list('custom_palette', palette, N=1000)
    q = pred.quantile(.99)
    norm = Normalize(vmin=np.min(pred), vmax=q)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = [to_hex(sm.to_rgba(p)) if p <= q else tailwind['red-800'] for p in pred]      
    return colors
df_state[f'pred_1min_color'] = _pred_color(df_state[f'pred_1min'], palette_pred)
df_state[f'score_1min_color'] = _pred_color(df_state[f'score_1min'], palette_score)

# =============================================================================
# Plot
# =============================================================================
def create_colorbar(ax, pred, palette, label='Value'):
    cmap = LinearSegmentedColormap.from_list('custom_palette', palette, N=1000)
    q = np.quantile(pred, 0.99)
    norm = Normalize(vmin=np.min(pred), vmax=q)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, aspect=10)
    cbar.set_label(label, labelpad=10)
    return cbar

for s in ['BTC', 'ETH']:
    df = df_state[df_state['symbol']==s]

    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Timeline Forecast on {s}USD Price (for 1min horizon)", pad=30)
    ax.set_ylabel(f"{s}USD Price", labelpad=20)
    ax.plot(df['_time'], df['price'].values, color=tailwind['stone-900'], alpha=.1, label=f"ETHUSD Price")
    ax.scatter(df['_time'], df['price'], color=df[f'pred_1min_color'], alpha=.9, s=4, label=f"Forecast")
    create_colorbar(ax, df['pred_1min'], palette_pred, label="Forecast Value")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Timeline Forecast 1min horizon.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Timeline Scores on {s}USD Price (for 1min horizon)", pad=30)
    ax.set_ylabel(f"{s}USD Price", labelpad=20)
    ax.plot(df['_time'], df['price'].values, color=tailwind['stone-900'], alpha=.1, label=f"ETHUSD Price")
    ax.scatter(df['_time'], df['price'], color=df[f'score_1min_color'], alpha=.7, s=4, label=f"Forecast")
    create_colorbar(ax, df['pred_1min'], palette_score, label="Score Value")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Timeline Score 1min horizon.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Timeline Scores on {s}USD Price (for 1min horizon)", pad=30)
    ax.set_ylabel(f"{s}USD Price", labelpad=20)
    ax.plot(df['_time'], df['price'].values, color=tailwind['stone-900'], alpha=.1, label=f"ETHUSD Price")
    ax.scatter(df[df['extreme_highvol']]['_time'], 
               df[df['extreme_highvol']]['price'], 
               color=tailwind[f'red-500'], alpha=.9, s=4)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Timeline High Vol 1min horizon.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Scatter of Actual vs Forecast Volatility (for {s} and 1min horizon)", pad=30)
    ax.set_ylabel(f"Forecast Volatility", labelpad=20)
    ax.set_xlabel(f"Actual Volatility", labelpad=20)
    ax.scatter(df['std_1min'],df['pred_1min'], color=tailwind['teal-500'],alpha=.3, s=4)
    ax.set_ylim((0, df['std_1min'].quantile(.999)))
    ax.set_xlim((0, df['std_1min'].quantile(.999)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Scatter Actual vs Forecast 1min horizon.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"Scatter of Forecast vs Scores Volatility (for {s} and 1min horizon)", pad=30)
    ax.set_ylabel(f"Scores", labelpad=20)
    ax.set_xlabel(f"Forecast Volatility", labelpad=20)
    ax.scatter(df['pred_1min'],df['score_1min'], color=tailwind['cyan-500'],alpha=.3, s=4)
    ax.set_ylim((0, df['score_1min'].quantile(.999)))
    ax.set_xlim((0, df['std_1min'].quantile(.999)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Scatter Forecast vs Scores 1min horizon.png'), dpi=200)
    plt.show()
    fig.clf()









