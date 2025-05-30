import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pytz

# === Load and prepare data ===
df = pd.read_parquet("stock_parquet", engine="pyarrow")

# Convert to datetime if needed
df['event_time'] = pd.to_datetime(df['event_time'])

# Convert UTC to US Eastern Time (or your preferred US timezone)
df['event_time'] = df['event_time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

# Sort and clean
df = df.sort_values(by=["symbol", "event_time"])
df = df.dropna(subset=["event_time", "close", "high", "low", "symbol"])

# === Select top N companies (by data points) ===
top_symbols = df['symbol'].value_counts().nlargest(5).index.tolist()

# === Setup plot layout ===
num_symbols = len(top_symbols)
fig = plt.figure(figsize=(14, 3.5 * num_symbols))
gs = gridspec.GridSpec(num_symbols, 1, hspace=0.4)

# === Plot each company ===
for i, symbol in enumerate(top_symbols):
    ax = fig.add_subplot(gs[i])
    symbol_df = df[df['symbol'] == symbol]
    
    ax.plot(symbol_df['event_time'], symbol_df['close'], label='Close', color='steelblue', linewidth=1.5)
    ax.plot(symbol_df['event_time'], symbol_df['high'], label='High', color='green', linestyle='--', linewidth=1)
    ax.plot(symbol_df['event_time'], symbol_df['low'], label='Low', color='red', linestyle='--', linewidth=1)
    
    ax.set_title(f"{symbol}", fontsize=14, weight='bold')
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Format the x-axis for time (now in US time)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    if i < num_symbols - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (US Eastern)", fontsize=12)

    ax.legend(loc='upper left', fontsize=10)

# === Final layout tweaks ===
plt.suptitle("Stock Prices Over Time (6/5/2025)", fontsize=16, weight='bold', y=0.95)
plt.tight_layout()
plt.show()
