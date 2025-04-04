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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.metrics import explained_variance_score

# =============================================================================
nr_round = 4
folder = 'HAR'
df_std = pd.read_csv(os.path.join('..', 'data', f'df_std.csv'))
horizons_all = np.array([1, 5, 15] + [60, 240, 720, 1440, 4320, 10080, 20160])

# =============================================================================
# HAR
# =============================================================================
def _features_har(series, df_har, horizons_har):
    data = pd.DataFrame(series)
    for h_col in horizons_har: 
        data[f'h_{h_col}'] = df_har[f'std_{h_col}min'].shift(freq=f'{h_col}min')
    return data

data_results = []
symbols = ['ETH', 'BTC']
df_horizons = pd.DataFrame([
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 1, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 5, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 15, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 60, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 240, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 720, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 1440, },
    ])
for s in symbols:
    for _, r_horizons in df_horizons.iloc[:].iterrows():

        h = r_horizons['h']
        horizons_har = horizons_all[horizons_all >= h]
        dt_fr = pd.to_datetime(r_horizons['dt_fr']).tz_localize('UTC')
        dt_to = pd.to_datetime(r_horizons['dt_to']).tz_localize('UTC')
                
        df_s = df_std[df_std['symbol']==s]
        df_s['y'] = df_s[f'std_{h}min']
        df_s['_time'] = pd.to_datetime(df_s['_time'])
        df_s.index = pd.to_datetime(df_s['_time'])
        
        #df_s['_time'] = pd.to_datetime(df_s['_time'])
        df_s = df_s[['_time', 'y', 'price']+[f'std_{h_col}min' for h_col in horizons_har]].dropna()
        df_f = _features_har(df_s['y'], df_s, horizons_har).dropna()
        df = df_s[(df_s.index >= dt_fr) & (df_s.index < dt_to)]#.reset_index()
                    
        Y_test, Y_pred = [], []
        date_range = pd.date_range(start=dt_fr,  end=dt_to,  freq='MS')#.tz_localize('UTC')
        for i, date in enumerate(date_range[:-1]):
            X_train = df_f[df_f.index < date].drop(columns='y')
            y_train = df_f[df_f.index < date]['y']
            if df_f.index.max() >= date:
                X_test = df_f[(df_f.index >= date) & (df_f.index < date_range[i+1])].drop(columns='y')
                y_test = df_f[(df_f.index >= date) & (df_f.index < date_range[i+1])]['y']
            
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                    
                Y_test += list(y_test)
                Y_pred += list(y_pred)
        df['y_test'] = Y_test
        df['y_pred'] = Y_pred
                    
        out = {
               'Model': f"HAR",
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
        ax.set_title(f"HAR Timeline {s} Forecast Volatility on {s}USD Price (for {h}min horizon)", pad=20)
        ax.set_ylabel(f"Price", labelpad=20)
        ax2.set_ylabel(f"Volatility", labelpad=20)
        ax.plot(df.index, df['price'], color=tailwind['stone-950'], linewidth=1, alpha=.99, label=f"{s}USD Price")
        ax2.plot(df.index, df['y_test'], color=tailwind['emerald-500'], linewidth=1, alpha=.9, label=f"Actual")
        ax2.plot(df.index, df['y_pred'], color=tailwind['orange-500'], linewidth=1, alpha=.9, label=f"Forecast")
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
        fig.savefig(os.path.join('results', folder, f'HAR {s} {h}min horizon Timeline Forecast.png'), dpi=200)
        fig.clf()
        
        _style_white()
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_title(f"HAR Scatter {s} Forecast on Actual Volatility (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"Forcasted Volatility", labelpad=20)
        ax.set_xlabel(f"Actual Volatility", labelpad=20)
        ax.scatter(df['y_test'],df['y_pred'], color=tailwind['orange-500'],alpha=.3, s=2, label=f"Forecast")
        ax.set_ylim((df['y_test'].quantile(.005), df['y_test'].quantile(.995)))
        ax.set_xlim((df['y_test'].quantile(.005), df['y_test'].quantile(.995)))
        #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        #ax.legend(loc='upper left', framealpha=0.1)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'HAR {s} {h}min horizon Scatter Forecast.png'), dpi=200)
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
                'EV': np.round(explained_variance_score(y_test, y_pred), nr_round),
                'R2': np.round(r2_score(y_test, y_pred),nr_round),
            }]
        df_week = pd.DataFrame(data_week)
        
        _style_white()
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_title(f"HAR {s} Weekly Model Fit on R2 Measure (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"R2 Measure",labelpad=20)
        ax.bar(df_week['Week End Time'],df_week['R2'], color=tailwind['orange-400'],alpha=.9)
        ax.set_ylim((0, 1))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'HAR {s} {h}min horizon Weekly R2.png'), dpi=200)
        fig.clf()
        
                    
        _style_white()
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_title(f"HAR {s} Weekly Model Fit on Correlation Measure (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"R2 Measure",labelpad=20)
        ax.bar(df_week['Week End Time'],df_week['Corr.'], color=tailwind['orange-300'],alpha=.9)
        ax.set_ylim((0, 1))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'HAR {s} {h}min horizon Weekly Corr.png'), dpi=200)
        fig.clf()  
        
df_results = pd.DataFrame(data_results)
df_results.to_csv(os.path.join('results', folder, f'df_results.csv'), index=False)

# =============================================================================
#             s = 'BTC'
#             lag = 4
#             h = 720
# =============================================================================