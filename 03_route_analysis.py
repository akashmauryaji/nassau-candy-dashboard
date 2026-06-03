import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/cleaned_data.csv')
os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

# ── 1. Route-Level Aggregation ────────────────────────────────────────────────
route_stats = df.groupby(['Route', 'Factory', 'Region']).agg(
    Total_Orders      = ('Order ID',           'count'),
    Avg_Lead_Time     = ('Lead Time (Days)',    'mean'),
    Std_Lead_Time     = ('Lead Time (Days)',    'std'),
    Min_Lead_Time     = ('Lead Time (Days)',    'min'),
    Max_Lead_Time     = ('Lead Time (Days)',    'max'),
    Total_Sales       = ('Sales',              'sum'),
    Avg_Profit_Margin = ('Profit Margin (%)',  'mean'),
).reset_index()

route_stats = route_stats.round(2)

# ── 2. Delay Frequency ────────────────────────────────────────────────────────
# Define delay threshold = mean lead time + 1 std deviation
threshold = df['Lead Time (Days)'].mean() + df['Lead Time (Days)'].std()
print(f"Delay Threshold: {threshold:.1f} days")

df['Is_Delayed'] = df['Lead Time (Days)'] > threshold

delay_by_route = df.groupby('Route').agg(
    Total_Orders  = ('Order ID',    'count'),
    Delayed_Orders= ('Is_Delayed',  'sum')
).reset_index()

delay_by_route['Delay Rate (%)'] = (
    delay_by_route['Delayed_Orders'] / delay_by_route['Total_Orders'] * 100
).round(2)

route_stats = route_stats.merge(delay_by_route[['Route', 'Delay Rate (%)']], on='Route', how='left')

# ── 3. Route Efficiency Score (0–100) ─────────────────────────────────────────
# Lower lead time = higher score
min_lt = route_stats['Avg_Lead_Time'].min()
max_lt = route_stats['Avg_Lead_Time'].max()

route_stats['Efficiency Score'] = (
    100 * (max_lt - route_stats['Avg_Lead_Time']) / (max_lt - min_lt)
).round(2)

# ── 4. Top & Bottom Routes ────────────────────────────────────────────────────
# Filter routes with at least 5 orders for reliability
reliable = route_stats[route_stats['Total_Orders'] >= 5]

top10    = reliable.nlargest(10,  'Efficiency Score')
bottom10 = reliable.nsmallest(10, 'Efficiency Score')

print("\n===== TOP 10 MOST EFFICIENT ROUTES =====")
print(top10[['Route', 'Avg_Lead_Time', 'Efficiency Score', 'Total_Orders', 'Delay Rate (%)']].to_string(index=False))

print("\n===== BOTTOM 10 LEAST EFFICIENT ROUTES =====")
print(bottom10[['Route', 'Avg_Lead_Time', 'Efficiency Score', 'Total_Orders', 'Delay Rate (%)']].to_string(index=False))

# ── 5. Plot: Top 10 vs Bottom 10 Efficiency Score ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.barplot(data=top10, x='Efficiency Score', y='Route', ax=axes[0],
            hue='Route', palette='YlGn', legend=False)
axes[0].set_title('Top 10 Most Efficient Routes', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Efficiency Score (0-100)')
axes[0].set_ylabel('Route')

sns.barplot(data=bottom10, x='Efficiency Score', y='Route', ax=axes[1],
            hue='Route', palette='OrRd', legend=False)
axes[1].set_title('Bottom 10 Least Efficient Routes', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Efficiency Score (0-100)')
axes[1].set_ylabel('Route')

plt.tight_layout()
plt.savefig('plots/11_top_bottom_routes.png', dpi=150)
plt.close()
print("\n[Saved] 11_top_bottom_routes.png")

# ── 6. State-Level Bottleneck Analysis ───────────────────────────────────────
state_stats = df.groupby('State/Province').agg(
    Total_Orders  = ('Order ID',          'count'),
    Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
    Delay_Rate    = ('Is_Delayed',         'mean'),
    Total_Sales   = ('Sales',             'sum'),
).reset_index()

state_stats['Delay_Rate (%)'] = (state_stats['Delay_Rate'] * 100).round(2)
state_stats['Avg_Lead_Time']  = state_stats['Avg_Lead_Time'].round(1)

# High volume + high delay = bottleneck
state_stats['Bottleneck Score'] = (
    state_stats['Total_Orders'] * state_stats['Delay_Rate (%)']
).round(2)

bottlenecks = state_stats.nlargest(10, 'Bottleneck Score')

print("\n===== TOP 10 GEOGRAPHIC BOTTLENECKS =====")
print(bottlenecks[['State/Province', 'Total_Orders', 'Avg_Lead_Time', 'Delay_Rate (%)', 'Bottleneck Score']].to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(data=bottlenecks, x='Bottleneck Score', y='State/Province',
            hue='State/Province', palette='Reds_r', legend=False)
plt.title('Top 10 Geographic Bottlenecks\n(High Volume + High Delay Rate)', fontsize=13, fontweight='bold')
plt.xlabel('Bottleneck Score (Orders × Delay Rate)')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('plots/12_geographic_bottlenecks.png', dpi=150)
plt.close()
print("[Saved] 12_geographic_bottlenecks.png")

# ── 7. Ship Mode Performance ──────────────────────────────────────────────────
shipmode_stats = df.groupby('Ship Mode').agg(
    Total_Orders  = ('Order ID',          'count'),
    Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
    Avg_Sales     = ('Sales',             'mean'),
    Avg_Margin    = ('Profit Margin (%)', 'mean'),
    Delay_Rate    = ('Is_Delayed',         'mean'),
).reset_index()

shipmode_stats['Delay_Rate (%)'] = (shipmode_stats['Delay_Rate'] * 100).round(2)
shipmode_stats = shipmode_stats.round(2)

print("\n===== SHIP MODE PERFORMANCE =====")
print(shipmode_stats.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=shipmode_stats, x='Ship Mode', y='Avg_Lead_Time', ax=axes[0],
            hue='Ship Mode', palette='Blues', legend=False)
axes[0].set_title('Avg Lead Time by Ship Mode')
axes[0].set_ylabel('Days')
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(data=shipmode_stats, x='Ship Mode', y='Avg_Sales', ax=axes[1],
            hue='Ship Mode', palette='Greens', legend=False)
axes[1].set_title('Avg Sales by Ship Mode')
axes[1].set_ylabel('Sales ($)')
axes[1].tick_params(axis='x', rotation=15)

sns.barplot(data=shipmode_stats, x='Ship Mode', y='Delay_Rate (%)', ax=axes[2],
            hue='Ship Mode', palette='Oranges', legend=False)
axes[2].set_title('Delay Rate by Ship Mode')
axes[2].set_ylabel('Delay Rate (%)')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('plots/13_shipmode_performance.png', dpi=150)
plt.close()
print("[Saved] 13_shipmode_performance.png")

# ── 8. Factory Performance ────────────────────────────────────────────────────
factory_stats = df.groupby('Factory').agg(
    Total_Orders  = ('Order ID',          'count'),
    Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
    Delay_Rate    = ('Is_Delayed',         'mean'),
    Total_Sales   = ('Sales',             'sum'),
    Avg_Margin    = ('Profit Margin (%)', 'mean'),
).reset_index()

factory_stats['Delay_Rate (%)'] = (factory_stats['Delay_Rate'] * 100).round(2)
factory_stats = factory_stats.round(2)

print("\n===== FACTORY PERFORMANCE =====")
print(factory_stats.to_string(index=False))

# ── 9. Region vs Factory Heatmap ──────────────────────────────────────────────
pivot = df.pivot_table(
    values='Lead Time (Days)',
    index='Factory',
    columns='Region',
    aggfunc='mean'
).round(1)

plt.figure(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5)
plt.title('Avg Lead Time (Days): Factory vs Region', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/14_factory_region_heatmap.png', dpi=150)
plt.close()
print("[Saved] 14_factory_region_heatmap.png")

# ── 10. Save All Results ──────────────────────────────────────────────────────
route_stats.to_csv('data/route_analysis.csv',     index=False)
state_stats.to_csv('data/state_analysis.csv',     index=False)
shipmode_stats.to_csv('data/shipmode_analysis.csv', index=False)
factory_stats.to_csv('data/factory_analysis.csv', index=False)

print("\n[Saved] All analysis CSVs to data/")
print("\nRoute Analysis Complete!")
