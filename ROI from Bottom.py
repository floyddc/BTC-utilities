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
btc = get_crypto_data("2013-01-01", current_date)
print(f"Data available from {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"Total days: {len(btc)}")

# ---------------------------- Cycles (bottom_date, peak_date) ----------------------------
cycles = {
    "Cycle 1 (2015)": ("2015-01-14", "2017-12-17"),
    "Cycle 2 (2018)": ("2018-12-16", "2021-11-08"),
    "Cycle 3 (2022)": ("2022-11-09", "2025-10-06"),
}

plt.style.use('dark_background')
plt.figure(figsize=(14, 7))
plt.gcf().patch.set_facecolor('#0a0a0a')
plt.gca().set_facecolor('#0a0a0a')

# ---------------------------- ROI calculation ----------------------------
colors = ['#ef4444', "#eae308", '#22c55e', '#3b82f6']
peak_rois = []  # Store final ROI values for y-axis ticks

for idx, (cycle_name, (bottom_date_str, peak_date_str)) in enumerate(cycles.items()):
    bottom_date = pd.Timestamp(bottom_date_str)
    
    if bottom_date not in btc.index:
        idx_nearest = btc.index.get_indexer([bottom_date], method='nearest')[0]
        bottom_date = btc.index[idx_nearest]
    
    bottom_price = btc.loc[bottom_date, 'Close']
    if isinstance(bottom_price, pd.Series):
        bottom_price = bottom_price.iloc[0]
    
    print(f"\n{cycle_name}")
    print(f"  📉 Bottom: ${bottom_price:,.2f} at {bottom_date.date()}")
    
    if peak_date_str is None:
        end_date = btc.index[-1]
    else:
        end_date = pd.Timestamp(peak_date_str)
        if end_date not in btc.index:
            idx_nearest = btc.index.get_indexer([end_date], method='nearest')[0]
            end_date = btc.index[idx_nearest]
    
    after_bottom = btc.loc[bottom_date:end_date].copy()
    
    if len(after_bottom) > 0:
        after_bottom = after_bottom.reset_index(drop=True)
        after_bottom['ROI'] = after_bottom['Close'] / bottom_price
        after_bottom['Days'] = range(len(after_bottom))
        
        # Remove outlier
        roi_values = after_bottom['ROI'].copy()
        
        rolling_median = roi_values.rolling(window=5, center=True, min_periods=1).median()
        rolling_std = roi_values.rolling(window=5, center=True, min_periods=1).std()
        
        # Identify outliers
        outliers = np.abs(roi_values - rolling_median) > (3 * rolling_std)
        
        # Outlier substitution
        roi_values[outliers] = np.nan
        after_bottom['ROI'] = roi_values.interpolate(method='linear')
        
        max_day = len(after_bottom) - 1
        max_price = after_bottom.loc[max_day, 'Close']
        if isinstance(max_price, pd.Series):
            max_price = max_price.iloc[0]
        max_roi = (max_price / bottom_price - 1) * 100
        
        print(f"  📈 Peak: +{max_roi:.1f}% (${max_price:,.2f}) after {int(max_day)} days")
        
        # Store final ROI value for y-axis
        final_roi = after_bottom['ROI'].iloc[-1]
        peak_rois.append(final_roi)
        
        plt.plot(
            after_bottom['Days'],
            after_bottom['ROI'],
            label=cycle_name,
            color=colors[idx % len(colors)],
            linewidth=1.5,
            alpha=0.9
        )

# ---------------------------- Styling ----------------------------
def format_roi(value, pos):
    """Converte moltiplicatore in percentuale"""
    return f'+{(value - 1) * 100:.0f}%' if value >= 1 else f'{(value - 1) * 100:.0f}%'

plt.yscale('log')
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_roi))

current_ticks = plt.gca().get_yticks()
custom_ticks = sorted(set(list(current_ticks) + peak_rois))
plt.yticks(custom_ticks)

plt.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.3)
plt.title("Bitcoin: ROI Bottom -> ATH", fontsize=18, fontweight='bold', color='white', pad=20)
plt.xlabel("Days from Bottom", fontsize=13, color='white')
plt.ylabel("ROI % from Bottom", fontsize=13, color='white')
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.15, color='gray', linestyle='-', linewidth=0.5)

max_roi = max(peak_rois)
plt.ylim(0.98, max_roi * 1.15)

plt.tight_layout()

print("\n" + "="*60)
print("📊 Chart generated!")
plt.show()
