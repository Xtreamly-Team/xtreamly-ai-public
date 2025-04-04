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

color_menu = tailwind['stone-800']
def _tbl(*arg):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     cellLoc='center', loc='center')
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(tailwind['stone-200'])
        cell.set_linewidth(1)
        cell.set_height(.2)
        if row == 0:
            cell.set_height(.25)
            cell.set_text_props(weight='semibold', color='white')
            cell.set_facecolor(color_menu)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    plt.show()
    return fig

# =============================================================================
# ARIMA
# =============================================================================
lags = [4,10,20]
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
# =============================================================================
# from PIL import Image
# size_0 = 700
# size_1 = 1600
# for lag in lags:
#     for s in symbols:
#         image = Image.new("RGB",(1600*4, 700*len(df_horizons)), (250,250,250))
#         for i,h in enumerate(df_horizons['h']):
#             names = [
#                 f'ARIMA {s} {h}min horizon {lag}lag Timeline Forecast.png',
#                 f'ARIMA {s} {h}min horizon {lag}lag Scatter Forecast.png',
#                 f'ARIMA {s} {h}min horizon {lag}lag Weekly R2.png',
#                 f'ARIMA {s} {h}min horizon {lag}lag Weekly Corr.png'
#                 ]
#             for j,n in enumerate(names):
#                 im = Image.open(os.path.join('results', 'ARIMA', n))
#                 im = im.resize((im.size[0] // 2, im.size[1] // 2))
#                 image.paste(im,(int(j*size_1), int(i*size_0)))
#         image.save(os.path.join('results', 'ARIMA', f'_{s} {lag}periods.png'))
# =============================================================================


fig, ax = plt.subplots(figsize=(12, 4))

df = pd.read_csv(os.path.join('results', 'ARIMA', 'df_results.csv'))
df = df.sort_values(['Model', 'Symbol', 'Horizon'])
df = df[df['Model'] == 'ARIMA 10 periods']
df['Model'] = df['Model'].replace(r'\s*10 periods', '', regex=True)
df['Correlation'] = df['Correlation'].apply(lambda x: f"{x:.2%}")
df['EV'] = df['EV'].apply(lambda x: f"{x:.2%}")
df['R2'] = df['R2'].apply(lambda x: f"{x:.2%}")

fig = _tbl()
fig.savefig(os.path.join('results', 'ARIMA', f'_KPI_ARIMA.png'), bbox_inches='tight', dpi=300)
     


# =============================================================================
# HAR
# =============================================================================
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
# =============================================================================
# from PIL import Image
# size_0 = 700
# size_1 = 1600
# for lag in lags:
#     for s in symbols:
#         image = Image.new("RGB",(1600*4, 700*len(df_horizons)), (250,250,250))
#         for i,h in enumerate(df_horizons['h']):
#             names = [
#                 f'HAR {s} {h}min horizon Timeline Forecast.png',
#                 f'HAR {s} {h}min horizon Scatter Forecast.png',
#                 f'HAR {s} {h}min horizon Weekly R2.png',
#                 f'HAR {s} {h}min horizon Weekly Corr.png'
#                 ]
#             for j,n in enumerate(names):
#                 im = Image.open(os.path.join('results', 'HAR', n))
#                 im = im.resize((im.size[0] // 2, im.size[1] // 2))
#                 image.paste(im,(int(j*size_1), int(i*size_0)))
#         image.save(os.path.join('results', 'HAR', f'_{s}.png'))
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 4))
df = pd.read_csv(os.path.join('results', 'HAR', 'df_results.csv'))
df = df.sort_values(['Model', 'Symbol', 'Horizon'])
df['Correlation'] = df['Correlation'].apply(lambda x: f"{x:.2%}")
df['EV'] = df['EV'].apply(lambda x: f"{x:.2%}")
df['R2'] = df['R2'].apply(lambda x: f"{x:.2%}")
fig = _tbl()
fig.savefig(os.path.join('results', 'HAR', f'_KPI_HAR.png'), bbox_inches='tight', dpi=300)

# =============================================================================
# ARCH
# =============================================================================
Models = ['ARCH', 'GARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'GJRGARCH']
symbols = ['ETH', 'BTC']
df_horizons = pd.DataFrame([
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 60, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 240, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 720, },
    {'dt_fr': '2024-10-01', 'dt_to': '2025-04-01', 'h': 1440, },
    ])
# =============================================================================
# from PIL import Image
# size_0 = 700
# size_1 = 1600
# for model in Models:
#     for s in symbols:
#         image = Image.new("RGB",(1600*4, 700*len(df_horizons)), (250,250,250))
#         for i,h in enumerate(df_horizons['h']):
#             names = [
#                 f'{model} {s} {h}min horizon Timeline Forecast.png',
#                 f'{model} {s} {h}min horizon Scatter Forecast.png',
#                 f'{model} {s} {h}min horizon Weekly R2.png',
#                 f'{model} {s} {h}min horizon Weekly Corr.png'
#                 ]
#             for j,n in enumerate(names):
#                 im = Image.open(os.path.join('results', 'ARCH', n))
#                 im = im.resize((im.size[0] // 2, im.size[1] // 2))
#                 image.paste(im,(int(j*size_1), int(i*size_0)))
#         image.save(os.path.join('results', 'ARCH', f'_{s} {model}.png'))
# =============================================================================
df_models = pd.read_csv(os.path.join('results', 'ARCH', 'df_results.csv'))
df_models = df_models.sort_values(['Symbol', 'Model', 'Horizon'])

for model in Models:
    fig, ax = plt.subplots(figsize=(12, 6))
    df = df_models[df_models['Model']==model].copy()
    df['Correlation'] = df['Correlation'].apply(lambda x: f"{x:.2%}")
    df['EV'] = df['EV'].apply(lambda x: f"{x:.2%}")
    df['R2'] = df['R2'].apply(lambda x: f"{x:.2%}")
    fig = _tbl()
    fig.savefig(os.path.join('results', 'ARCH', f'_KPI_{model}.png'), bbox_inches='tight', dpi=300)











