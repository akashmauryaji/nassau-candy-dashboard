import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/cleaned_data.csv')
os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Dataset shape:", df.shape)
print("\nColumn list:", list(df.columns))
print("\nBasic Stats:")
print(df[['Lead Time (Days)', 'Sales', 'Gross Profit', 'Profit Margin (%)']].describe().round(2))

# ── Plot 1: Lead Time Distribution ────────────────────────────────────────────
plt.figure()
sns.histplot(df['Lead Time (Days)'], bins=30, color='steelblue', kde=True)
plt.title('Distribution of Shipping Lead Time (Days)')
plt.xlabel('Lead Time (Days)')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('plots/01_lead_time_distribution.png', dpi=150)
plt.close()
print("\n[Saved] 01_lead_time_distribution.png")

# ── Plot 2: Avg Lead Time by Region ───────────────────────────────────────────
region_lt = df.groupby('Region')['Lead Time (Days)'].mean().sort_values(ascending=False).reset_index()
plt.figure()
sns.barplot(data=region_lt, x='Region', y='Lead Time (Days)', palette='Blues_d')
plt.title('Average Lead Time by Region')
plt.xlabel('Region')
plt.ylabel('Avg Lead Time (Days)')
plt.tight_layout()
plt.savefig('plots/02_lead_time_by_region.png', dpi=150)
plt.close()
print("[Saved] 02_lead_time_by_region.png")

# ── Plot 3: Avg Lead Time by Ship Mode ────────────────────────────────────────
mode_lt = df.groupby('Ship Mode')['Lead Time (Days)'].mean().sort_values(ascending=False).reset_index()
plt.figure()
sns.barplot(data=mode_lt, x='Ship Mode', y='Lead Time (Days)', palette='Oranges_d')
plt.title('Average Lead Time by Ship Mode')
plt.xlabel('Ship Mode')
plt.ylabel('Avg Lead Time (Days)')
plt.tight_layout()
plt.savefig('plots/03_lead_time_by_shipmode.png', dpi=150)
plt.close()
print("[Saved] 03_lead_time_by_shipmode.png")

# ── Plot 4: Avg Lead Time by Factory ──────────────────────────────────────────
factory_lt = df.groupby('Factory')['Lead Time (Days)'].mean().sort_values(ascending=False).reset_index()
plt.figure()
sns.barplot(data=factory_lt, x='Lead Time (Days)', y='Factory', palette='Greens_d')
plt.title('Average Lead Time by Factory')
plt.xlabel('Avg Lead Time (Days)')
plt.ylabel('Factory')
plt.tight_layout()
plt.savefig('plots/04_lead_time_by_factory.png', dpi=150)
plt.close()
print("[Saved] 04_lead_time_by_factory.png")

# ── Plot 5: Top 10 Most Efficient Routes ──────────────────────────────────────
route_stats = df.groupby('Route').agg(
    Avg_Lead_Time   = ('Lead Time (Days)', 'mean'),
    Total_Orders    = ('Order ID', 'count'),
    Avg_Profit_Margin = ('Profit Margin (%)', 'mean')
).reset_index()
route_stats['Avg_Lead_Time'] = route_stats['Avg_Lead_Time'].round(1)

top10 = route_stats.nsmallest(10, 'Avg_Lead_Time')
bot10 = route_stats.nlargest(10, 'Avg_Lead_Time')

plt.figure(figsize=(12, 6))
sns.barplot(data=top10, x='Avg_Lead_Time', y='Route', palette='YlGn_r')
plt.title('Top 10 Most Efficient Routes (Lowest Lead Time)')
plt.xlabel('Avg Lead Time (Days)')
plt.ylabel('Route')
plt.tight_layout()
plt.savefig('plots/05_top10_efficient_routes.png', dpi=150)
plt.close()
print("[Saved] 05_top10_efficient_routes.png")

# ── Plot 6: Bottom 10 Least Efficient Routes ──────────────────────────────────
plt.figure(figsize=(12, 6))
sns.barplot(data=bot10, x='Avg_Lead_Time', y='Route', palette='OrRd')
plt.title('Bottom 10 Least Efficient Routes (Highest Lead Time)')
plt.xlabel('Avg Lead Time (Days)')
plt.ylabel('Route')
plt.tight_layout()
plt.savefig('plots/06_bottom10_routes.png', dpi=150)
plt.close()
print("[Saved] 06_bottom10_routes.png")

# ── Plot 7: Sales by Division ─────────────────────────────────────────────────
div_sales = df.groupby('Division')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(data=div_sales, x='Division', y='Sales', palette='Purples_d')
plt.title('Total Sales by Division')
plt.xlabel('Division')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('plots/07_sales_by_division.png', dpi=150)
plt.close()
print("[Saved] 07_sales_by_division.png")

# ── Plot 8: Monthly Order Trend ───────────────────────────────────────────────
monthly = df.groupby(['Order Year', 'Order Month'])['Order ID'].count().reset_index()
monthly['Period'] = monthly['Order Year'].astype(str) + '-' + monthly['Order Month'].astype(str).str.zfill(2)
monthly = monthly.sort_values('Period')

plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly, x='Period', y='Order ID', marker='o', color='steelblue')
plt.title('Monthly Order Volume Trend')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/08_monthly_order_trend.png', dpi=150)
plt.close()
print("[Saved] 08_monthly_order_trend.png")

# ── Plot 9: Heatmap - Lead Time by Region vs Ship Mode ────────────────────────
pivot = df.pivot_table(values='Lead Time (Days)', index='Region', columns='Ship Mode', aggfunc='mean').round(1)
plt.figure(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Avg Lead Time (Days): Region vs Ship Mode')
plt.tight_layout()
plt.savefig('plots/09_heatmap_region_shipmode.png', dpi=150)
plt.close()
print("[Saved] 09_heatmap_region_shipmode.png")

# ── Plot 10: Profit Margin by Ship Mode ───────────────────────────────────────
plt.figure()
sns.boxplot(data=df, x='Ship Mode', y='Profit Margin (%)', palette='Set2')
plt.title('Profit Margin Distribution by Ship Mode')
plt.xlabel('Ship Mode')
plt.ylabel('Profit Margin (%)')
plt.tight_layout()
plt.savefig('plots/10_profit_margin_by_shipmode.png', dpi=150)
plt.close()
print("[Saved] 10_profit_margin_by_shipmode.png")

# ── Print Key Insights ────────────────────────────────────────────────────────
print("\n========== KEY INSIGHTS ==========")
print(f"\nTotal Orders        : {len(df):,}")
print(f"Total Sales         : ${df['Sales'].sum():,.2f}")
print(f"Total Gross Profit  : ${df['Gross Profit'].sum():,.2f}")
print(f"Avg Lead Time       : {df['Lead Time (Days)'].mean():.1f} days")
print(f"Avg Profit Margin   : {df['Profit Margin (%)'].mean():.1f}%")

print("\n-- Avg Lead Time by Region --")
print(region_lt.to_string(index=False))

print("\n-- Avg Lead Time by Ship Mode --")
print(mode_lt.to_string(index=False))

print("\n-- Top 5 Most Efficient Routes --")
print(top10[['Route', 'Avg_Lead_Time', 'Total_Orders']].head(5).to_string(index=False))

print("\n-- Top 5 Least Efficient Routes --")
print(bot10[['Route', 'Avg_Lead_Time', 'Total_Orders']].head(5).to_string(index=False))

# ── Save Route Stats ──────────────────────────────────────────────────────────
route_stats.to_csv('data/route_stats.csv', index=False)
print("\n[Saved] data/route_stats.csv")
print("\nEDA Complete!")
