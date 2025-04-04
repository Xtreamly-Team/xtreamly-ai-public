# %reset -f
import os
import sys
import pandas as pd
import numpy as np
import time
import pytz
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sklearn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
pd.set_option('display.max_columns', None)
load_dotenv()
from settings.plot import tailwind, _style, _style_white
from gcp.func import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.metrics import explained_variance_score

# =============================================================================
nr_round = 4
folder = 'ARCH'
df_std = pd.read_csv(os.path.join('..', 'data', f'df_std.csv'))
df_std['_time'] = pd.to_datetime(df_std['_time'])
horizons_arch = [60, 240, 720, 1440]

# =============================================================================
# ARCH
# =============================================================================
Models = ['ARCH', 'GARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'GJRGARCH']

data_results = []
symbols = ['ETH', 'BTC']
df_horizons = pd.DataFrame([
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 60, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 240, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 720, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 1440, },
    ])
from arch import arch_model
for s in symbols: 
    for _, r_horizons in df_horizons.iloc[:].iterrows():
        h = r_horizons['h']
        dt_fr = pd.to_datetime(r_horizons['dt_fr']).tz_localize('UTC')
        dt_to = pd.to_datetime(r_horizons['dt_to']).tz_localize('UTC')
        
        df_s = df_std[(df_std['symbol']==s) & (df_std['_time'] < dt_to)].copy()
        df_s['ret'] = df_s[f'ret_{h}min']
        df_s['std'] = df_s[f'std_{h}min']
        df_s.index = df_s['_time']
        
        returns_hmin = np.log(df_s[f'price'] / df_s[f'price'].shift(freq=f'{h}min')).dropna() * 100  # percent
        returns_1min = np.log(df_s[f'price'] / df_s[f'price'].shift(freq=f'1min')).dropna()
        returns_1min = returns_1min.loc[returns_hmin.index]
        std_hmin = (returns_1min.rolling(h).std() * np.sqrt(h)).dropna()
        returns_hmin = returns_hmin.loc[std_hmin.index]
        
        for model_name in Models: 
            # model_name = 'GJRGARCH'
            if model_name == 'GJRGARCH':
                model = arch_model(returns_hmin, vol='GARCH', p=1, q=1, o=1, dist='normal', rescale=False)
            elif model_name == "ARCH":
                model = arch_model(returns_hmin, vol=model_name, p=1, rescale=False)
            else:
                model= arch_model(returns_hmin, vol=model_name, p=1, q=1, rescale=False)

            res = model.fit(last_obs=dt_fr, disp='off')
            forecasts = res.forecast(horizon=1, start=dt_fr, method='simulation')
            y_pred = forecasts.variance/100
            y_test = std_hmin.loc[y_pred.index]
            
            df = df_s.loc[y_pred.index]
            df['y_test'] = y_test
            df['y_pred'] = y_pred
            y_test = df['y_test'].values
            y_pred = df['y_pred'].values
                     
            out = {
                   'Model': f"{model_name}",
                   'Symbol': s,
                   'Horizon': h,
                   'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
                   'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
                   'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
                   'Correlation': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
                   'EV': max(0, np.round(explained_variance_score(y_test, y_pred), nr_round)),
                   'R2': max(0, np.round(r2_score(y_test, y_pred),nr_round)),
                   }
            print(out)
            data_results += [out]
            # =============================================================================
            # Plot
            # =============================================================================
            _style_white()
            fig, ax = plt.subplots(figsize=(16, 7))
            ax2 = ax.twinx()
            ax.set_title(f"{model_name} Timeline {s} Forecast Volatility on {s}USD Price (for {h}min horizon)", pad=20)
            ax.set_ylabel(f"Price", labelpad=20)
            ax2.set_ylabel(f"Volatility (up to 99.99% quantile)", labelpad=20)
            ax.plot(df.index, df['price'], color=tailwind['stone-950'], linewidth=1, alpha=.99, label=f"{s}USD Price")
            ax2.plot(df.index, df['y_test'], color=tailwind['emerald-500'], linewidth=1, alpha=.8, label=f"Actual")
            ax2.plot(df.index, df['y_pred'], color=tailwind['yellow-500'], linewidth=1, alpha=.8, label=f"Forecast")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
            ax2.set_ylim((0, df['y_test'].quantile(.9999)))
            ax.set_yticks(ax.get_yticks())
            ax.set_xticks(ax.get_xticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
            ax2.grid(True, linestyle='-', linewidth=1, alpha=0.0)
            ax.legend(loc='upper left', framealpha=0.1)
            ax2.legend(loc='upper right', framealpha=0.1)
            for spine in ax.spines.values(): spine.set_visible(False)
            for spine in ax2.spines.values(): spine.set_visible(False)
            fig.tight_layout(rect=[0.004, 0.004, .996, .996])
            fig.savefig(os.path.join('results', folder, f'{model_name} {s} {h}min horizon Timeline Forecast.png'), dpi=200)
            fig.clf()
            
            _style_white()
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.set_title(f"{model_name} Scatter {s} Forecast on Actual Volatility (for {h}min horizon)", pad=30)
            ax.set_ylabel(f"Forcasted Volatility (up to 99.5% quantile)", labelpad=20)
            ax.set_xlabel(f"Actual Volatility (up to 99.5% quantile)", labelpad=20)
            ax.scatter(df['y_test'],df['y_pred'], color=tailwind['yellow-500'],alpha=.3, s=2, label=f"Forecast")
            ax.set_ylim((.0, df['y_pred'].quantile(.995)))
            ax.set_xlim((.0, df['y_test'].quantile(.995)))
            #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
            ax.set_yticks(ax.get_yticks())
            ax.set_xticks(ax.get_xticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
            #ax.legend(loc='upper left', framealpha=0.1)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.tight_layout(rect=[0.004, 0.004, .996, .996])
            fig.savefig(os.path.join('results', folder, f'{model_name} {s} {h}min horizon Scatter Forecast.png'), dpi=200)
            fig.clf()
            
            data_week = []
            df['week_number'] = df['_time'].dt.isocalendar().week
            
            for week in df['week_number'].unique():
                df_w = df[df['week_number'] == week]
                y_test = df_w['y_test'].values
                y_pred = df_w['y_pred'].values
                data_week += [{
                    'Week End Time': df_w['_time'].max().strftime('%Y-%m-%d'),
                    'Week Nr': week,
                    'Count': df_w.shape[0],
                    'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
                    'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
                    'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
                    'Corr.': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
                    'EV': max(0.0, np.round(explained_variance_score(y_test, y_pred), nr_round)),
                    'R2': max(0.0, np.round(r2_score(y_test, y_pred),nr_round)),
                }]
            df_week = pd.DataFrame(data_week)
            
            _style_white()
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.set_title(f"{model_name} {s} Weekly Model Fit on R2 Measure (for {h}min horizon)", pad=30)
            ax.set_ylabel(f"R2 Measure",labelpad=20)
            ax.bar(df_week['Week End Time'],df_week['R2'], color=tailwind['yellow-400'],alpha=.9)
            ax.set_ylim((0, 1))
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
            ax.set_yticks(ax.get_yticks())
            ax.set_xticks(ax.get_xticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.tight_layout(rect=[0.004, 0.004, .996, .996])
            fig.savefig(os.path.join('results', folder, f'{model_name} {s} {h}min horizon Weekly R2.png'), dpi=200)
            fig.clf()
            
            _style_white()
            fig, ax = plt.subplots(figsize=(16, 7))
            ax.set_title(f"{model_name} {s} Weekly Model Fit on Correlation Measure (for {h}min horizon)", pad=30)
            ax.set_ylabel(f"R2 Measure",labelpad=20)
            ax.bar(df_week['Week End Time'],df_week['Corr.'], color=tailwind['yellow-300'],alpha=.9)
            ax.set_ylim((0, 1))
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
            ax.set_yticks(ax.get_yticks())
            ax.set_xticks(ax.get_xticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.tight_layout(rect=[0.004, 0.004, .996, .996])
            fig.savefig(os.path.join('results', folder, f'{model_name} {s} {h}min horizon Weekly Corr.png'), dpi=200)
            fig.clf()    
        
df_results = pd.DataFrame(data_results)
df_results.to_csv(os.path.join('results', folder, f'df_results.csv'), index=False)

