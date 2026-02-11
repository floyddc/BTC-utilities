import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np
from matplotlib.ticker import FuncFormatter

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
btc = get_crypto_data("2012-01-01", current_date)
print(f"Data available from {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"Total days: {len(btc)}")

# ---------------------------- Cycles (halving to halving) ----------------------------
cycles = {
    "Halving 1-2": ("2012-11-28", "2016-07-09"),
    "Halving 2-3": ("2016-07-09", "2020-05-11"),
    "Halving 3-4": ("2020-05-11", "2024-04-19"),
    "Halving 4-5": ("2024-04-19", None),  
}

plt.style.use('dark_background')
plt.figure(figsize=(14, 7))
plt.gcf().patch.set_facecolor('#0a0a0a')
plt.gca().set_facecolor('#0a0a0a')

# ---------------------------- ROI calculation ----------------------------
colors = ['#ef4444', "#eae308", '#22c55e', '#3b82f6']
peak_rois = []  # Store final ROI values for y-axis ticks

for idx, (cycle_name, (halving_start_str, halving_end_str)) in enumerate(cycles.items()):
    halving_start = pd.Timestamp(halving_start_str)
    
    if halving_start not in btc.index:
        idx_nearest = btc.index.get_indexer([halving_start], method='nearest')[0]
        halving_start = btc.index[idx_nearest]
    
    start_price = btc.loc[halving_start, 'Close']
    if isinstance(start_price, pd.Series):
        start_price = start_price.iloc[0]
    
    print(f"\n{cycle_name}")
    print(f"  🔸 Halving Start: ${start_price:,.2f} at {halving_start.date()}")
    
    if halving_end_str is None:
        end_date = btc.index[-1]
    else:
        end_date = pd.Timestamp(halving_end_str)
        if end_date not in btc.index:
            idx_nearest = btc.index.get_indexer([end_date], method='nearest')[0]
            end_date = btc.index[idx_nearest]
    
    cycle_data = btc.loc[halving_start:end_date].copy()
    
    if len(cycle_data) > 0:
        cycle_data = cycle_data.reset_index(drop=True)
        cycle_data['ROI'] = cycle_data['Close'] / start_price
        cycle_data['Days'] = range(len(cycle_data))
        
        # Remove outlier
        roi_values = cycle_data['ROI'].copy()
        
        rolling_median = roi_values.rolling(window=5, center=True, min_periods=1).median()
        rolling_std = roi_values.rolling(window=5, center=True, min_periods=1).std()
        
        # Identify outliers
        outliers = np.abs(roi_values - rolling_median) > (3 * rolling_std)
        
        # Outlier substitution
        roi_values[outliers] = np.nan
        cycle_data['ROI'] = roi_values.interpolate(method='linear')
        
        max_day = len(cycle_data) - 1
        end_price = cycle_data.loc[max_day, 'Close']
        if isinstance(end_price, pd.Series):
            end_price = end_price.iloc[0]
        cycle_roi = (end_price / start_price - 1) * 100
        
        print(f"  🔹 Cycle End: +{cycle_roi:.1f}% (${end_price:,.2f}) after {int(max_day)} days")
        
        # Store final ROI value for y-axis
        final_roi = cycle_data['ROI'].iloc[-1]
        peak_rois.append(final_roi)
        
        plt.plot(
            cycle_data['Days'],
            cycle_data['ROI'],
            label=cycle_name,
            color=colors[idx % len(colors)],
            linewidth=1.5,
            alpha=0.9
        )

# ---------------------------- Styling ----------------------------
def format_roi(value, pos):
    return f'+{(value - 1) * 100:.0f}%' if value >= 1 else f'{(value - 1) * 100:.0f}%'

plt.yscale('log')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_roi))

current_ticks = plt.gca().get_yticks()
custom_ticks = sorted(set(list(current_ticks) + peak_rois))
plt.yticks(custom_ticks)

plt.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.3)
plt.title("Bitcoin: ROI from Halving to Halving", fontsize=18, fontweight='bold', color='white', pad=20)
plt.xlabel("Days from Halving", fontsize=13, color='white')
plt.ylabel("ROI % from Halving", fontsize=13, color='white')
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.15, color='gray', linestyle='-', linewidth=0.5)

max_roi = max(peak_rois)
plt.ylim(0.98, max_roi * 1.15)

plt.tight_layout()

print("\n" + "="*60)
print("📊 Chart generated!")
plt.show()
