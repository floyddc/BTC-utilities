import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np

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

btc = get_crypto_data("2013-01-01", "2026-02-10")
print(f"Data available from {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"Total days: {len(btc)}")

# ---------------------------- Cycles ----------------------------
cycles = {
    "Cycle 1 (2013)": ("2013-11-29", "2015-01-14"),
    "Cycle 2 (2017)": ("2017-12-17", "2018-12-15"),
    "Cycle 3 (2021)": ("2021-11-08", "2022-11-21"),
    "Cycle 4 (2025)": ("2025-10-06", None),
}

plt.style.use('dark_background')
plt.figure(figsize=(14, 7))
plt.gcf().patch.set_facecolor('#0a0a0a')
plt.gca().set_facecolor('#0a0a0a')

# ---------------------------- ROI calculation ----------------------------
colors = ['#ef4444', "#eae308", '#22c55e', '#3b82f6']

for idx, (cycle_name, (peak_date_str, bottom_date_str)) in enumerate(cycles.items()):
    peak_date = pd.Timestamp(peak_date_str)
    
    if peak_date not in btc.index:
        idx_nearest = btc.index.get_indexer([peak_date], method='nearest')[0]
        peak_date = btc.index[idx_nearest]
    
    peak_price = btc.loc[peak_date, 'Close']
    if isinstance(peak_price, pd.Series):
        peak_price = peak_price.iloc[0]
    
    print(f"\n{cycle_name}")
    print(f"  📈 ATH: ${peak_price:,.2f} at {peak_date.date()}")
    
    if bottom_date_str is None:
        end_date = btc.index[-1]
    else:
        end_date = pd.Timestamp(bottom_date_str)
        if end_date not in btc.index:
            idx_nearest = btc.index.get_indexer([end_date], method='nearest')[0]
            end_date = btc.index[idx_nearest]
    
    after_peak = btc.loc[peak_date:end_date].copy()
    
    if len(after_peak) > 0:
        after_peak = after_peak.reset_index(drop=True)
        after_peak['ROI'] = after_peak['Close'] / peak_price
        after_peak['Days'] = range(len(after_peak))
        
        # Remove outlier
        roi_values = after_peak['ROI'].copy()
        
        rolling_median = roi_values.rolling(window=5, center=True, min_periods=1).median()
        rolling_std = roi_values.rolling(window=5, center=True, min_periods=1).std()
        
        # Identify outliers
        outliers = np.abs(roi_values - rolling_median) > (3 * rolling_std)
        
        # Outlier substitution
        roi_values[outliers] = np.nan
        after_peak['ROI'] = roi_values.interpolate(method='linear')
        
        min_day = len(after_peak) - 1
        min_price = after_peak.loc[min_day, 'Close']
        if isinstance(min_price, pd.Series):
            min_price = min_price.iloc[0]
        min_roi = (min_price / peak_price - 1) * 100
        
        print(f"  📉 Bottom: {min_roi:.1f}% (${min_price:,.2f}) after {int(min_day)} days")
        
        plt.plot(
            after_peak['Days'],
            after_peak['ROI'],
            label=cycle_name,
            color=colors[idx % len(colors)],
            linewidth=1.5,
            alpha=0.9
        )

# ---------------------------- Styling ----------------------------
plt.axhline(0.2, color='gray', linewidth=0.5, linestyle=':', alpha=0.3)
plt.title("Bitcoin: ROI from ATH", fontsize=18, fontweight='bold', color='white', pad=20)
plt.xlabel("Days from ATH", fontsize=13, color='white')
plt.ylabel("ROI (1.0 = ATH)", fontsize=13, color='white')
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.15, color='gray', linestyle='-', linewidth=0.5)
plt.ylim(-0.1, 1.1)
plt.tight_layout()

print("\n" + "="*60)
print("📊 Chart generated!")
plt.show()
