import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------- Download data ----------------------------
print("Downloading BTC data from CryptoCompare...")

def get_crypto_data(start_date, end_date):
    """Scarica dati storici da CryptoCompare"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    
    all_data = []
    current_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    while current_ts < end_ts:
        params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': 2000,
            'toTs': min(current_ts + (2000 * 86400), end_ts)
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['Response'] == 'Success':
            all_data.extend(data['Data']['Data'])
            current_ts += 2000 * 86400
        else:
            print(f"Errore: {data}")
            break
    
    df = pd.DataFrame(all_data)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('Date')[['close']]
    df.columns = ['Close']
    df = df[~df.index.duplicated(keep='last')]
    
    return df

current_date = datetime.now().strftime("%Y-%m-%d")
btc = get_crypto_data("2013-01-01", current_date)
print(f"Data available from {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"Total days: {len(btc)}")

# ---------------------------- Calculate Risk Index ----------------------------
print("\nCalculating Risk Index...")

window_days = 730  

btc['Min_Rolling'] = btc['Close'].rolling(window=window_days, min_periods=180).min()
btc['Max_Rolling'] = btc['Close'].rolling(window=window_days, min_periods=180).max()

btc['Risk_Index'] = (btc['Close'] - btc['Min_Rolling']) / (btc['Max_Rolling'] - btc['Min_Rolling'])

btc['Risk_Index'] = btc['Risk_Index'].bfill().fillna(0.5)
btc['Risk_Index'] = btc['Risk_Index'].clip(0, 1)

btc['Risk_Index_Smooth'] = btc['Risk_Index'].rolling(window=7, center=True, min_periods=1).mean()

print(f"Risk Index range: {btc['Risk_Index_Smooth'].min():.3f} - {btc['Risk_Index_Smooth'].max():.3f}")

# ---------------------------- Create colored line plot ----------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')

points = np.array([btc.index, btc['Risk_Index_Smooth']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

dates_numeric = plt.matplotlib.dates.date2num(btc.index)
risk_values = btc['Risk_Index_Smooth'].values

points = np.array([dates_numeric, risk_values]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

colors_list = [
    '#3b82f6',  
    '#22c55e',  
    '#eae308',  
    '#f97316',  
    '#ef4444',  
]
n_bins = 100
cmap = LinearSegmentedColormap.from_list('risk', colors_list, N=n_bins)

lc = LineCollection(segments, cmap=cmap, linewidth=2.5, alpha=0.9)
lc.set_array(risk_values[:-1])
lc.set_clim(0, 1)

line = ax.add_collection(lc)

# ---------------------------- Styling ----------------------------
time_range = (btc.index[-1] - btc.index[0]).days
margin_days = pd.Timedelta(days=int(time_range * 0.02))
ax.set_xlim(btc.index[0], btc.index[-1] + margin_days)
ax.set_ylim(-0.05, 1.05)

ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

risk_levels = [0.2, 0.4, 0.6, 0.8]
for level in risk_levels:
    ax.axhline(level, color='gray', linewidth=0.5, linestyle=':', alpha=0.3)

ax.set_title("BTC Risk Index", fontsize=20, fontweight='bold', color='white', pad=20)
ax.set_xlabel("Date", fontsize=13, color='white')
ax.set_ylabel("Risk", fontsize=13, color='white')
ax.grid(alpha=0.1, color='gray', linestyle='-', linewidth=0.5)

fig.autofmt_xdate()
ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b \'%y'))

from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0, 0), 1, 1, fc='#3b82f6', label='0.0 - 0.2'),
    Rectangle((0, 0), 1, 1, fc='#22c55e', label='0.2 - 0.4'),
    Rectangle((0, 0), 1, 1, fc='#eae308', label='0.4 - 0.6'),
    Rectangle((0, 0), 1, 1, fc='#f97316', label='0.6 - 0.8'),
    Rectangle((0, 0), 1, 1, fc='#ef4444', label='0.8 - 1.0'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
          framealpha=0.9, ncol=5, bbox_to_anchor=(0, -0.08))

current_price = btc['Close'].iloc[-1]
current_risk = btc['Risk_Index_Smooth'].iloc[-1]
current_date_str = btc.index[-1].strftime('%d %b %Y')

print(f"\n{'='*60}")
print(f"📊 Current data ({current_date_str}):")
print(f"   BTC Price: ${current_price:,.0f}")
print(f"   Risk Index: {current_risk:.3f} ({current_risk*100:.1f}%)")

if current_risk < 0.2:
    risk_label = "🟦 Very Low"
elif current_risk < 0.4:
    risk_label = "🟩 Low"
elif current_risk < 0.6:
    risk_label = "🟨 Medium"
elif current_risk < 0.8:
    risk_label = "🟧 High"
else:
    risk_label = "🟥 Very High"

print(f"   Level: {risk_label}")
print(f"{'='*60}")

plt.tight_layout()
plt.show()
