# %reset -f
import os
import sys
import pandas as pd
import joblib
import time
import pytz
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
pd.set_option('display.max_columns', None)
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from gcp.func import *
from settings.plot import tailwind, _style

df_market_1sec = pd.read_csv(os.path.join('..', 'data', f'df_market_1sec.csv'))
data_std_1sec = []
horizons_1sec = [1, 5, 15]
for s in df_market_1sec['symbol'].unique():
    df_s = df_market_1sec[df_market_1sec['symbol']==s].copy()
    df_s.index = pd.to_datetime(df_s['_time'])
    df_s = df_s[['price']]
    for h in horizons_1sec:
        df_s[f'price_{h}min_old'] = df_s['price'].shift(freq=f'{h}min')
        df_s[f'ret_{h}min'] = np.log(df_s['price']/df_s[f'price_{h}min_old'])*100
        df_s[f'std_{h}min'] = df_s[f'ret_{h}min'].rolling(window=f'{h}min').std() 
    df_s = df_s[df_s.index.second == 0]
    df_s['symbol'] = s
    data_std_1sec += [df_s.reset_index()]
df_std_1sec = pd.concat(data_std_1sec)
df_std_1sec = df_std_1sec[['_time', 'symbol']+[f'std_{h}min' for h in horizons_1sec]]

df_market_1min = pd.read_csv(os.path.join('..', 'data', f'df_market_1min.csv'))
data_std_1min = []
horizons_1min = [60, 240, 720, 1440, 4320, 10080, 20160]
for s in df_market_1min['symbol'].unique():
    df_s = df_market_1min[df_market_1min['symbol']==s].copy()
    df_s.index = pd.to_datetime(df_s['_time'])
    df_s = df_s[['open']].rename(columns={'open':'price'})
    for h in horizons_1min:
        df_s[f'price_{h}min_old'] = df_s['price'].shift(freq=f'{h}min')
        df_s[f'ret_{h}min'] = np.log(df_s['price']/df_s[f'price_{h}min_old'])*100
        df_s[f'std_{h}min'] = df_s[f'ret_{h}min'].rolling(window=f'{h}min').std() 
    df_s = df_s[df_s.index.second == 0]
    df_s['symbol'] = s
    data_std_1min += [df_s.reset_index()]
df_std_1min = pd.concat(data_std_1min)
df_std_1min = df_std_1min[['_time', 'symbol', 'price']+[f'std_{h}min' for h in horizons_1min]+[f'ret_{h}min' for h in horizons_1min]]

horizons = horizons_1sec + horizons_1min
df_std = df_std_1min.merge(df_std_1sec, on=['_time','symbol'], how='left')
df_std = df_std[['_time', 'symbol', 'price']+[f'std_{h}min' for h in horizons]+[f'ret_{h}min' for h in horizons_1min]]
df_std.to_csv(os.path.join('..', 'data', f'df_std.csv'), index=False)



