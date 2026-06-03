import pandas as pd
import numpy as np

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_excel('data/Nassau_Candy_Distributor.xlsx')
print("Original shape:", df.shape)

# ── 2. Fix Ship Date (years are ~2 years ahead — shift back) ──────────────────
df['Ship Date'] = df['Ship Date'] - pd.DateOffset(years=2)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# ── 3. Calculate Shipping Lead Time ───────────────────────────────────────────
df['Lead Time (Days)'] = (df['Ship Date'] - df['Order Date']).dt.days

# Remove invalid lead times (negative or zero)
df = df[df['Lead Time (Days)'] > 0]
print("After removing invalid lead times:", df.shape)

# ── 4. Map Products to Factories ──────────────────────────────────────────────
product_factory_map = {
    'Wonka Bar - Nutty Crunch Surprise' : "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows'         : "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious'    : "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate'        : "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel' : "Wicked Choccy's",
    'Laffy Taffy'                       : 'Sugar Shack',
    'SweeTARTS'                         : 'Sugar Shack',
    'Nerds'                             : 'Sugar Shack',
    'Fun Dip'                           : 'Sugar Shack',
    'Fizzy Lifting Drinks'              : 'Sugar Shack',
    'Everlasting Gobstopper'            : 'Secret Factory',
    'Lickable Wallpaper'                : 'Secret Factory',
    'Wonka Gum'                         : 'Secret Factory',
    'Hair Toffee'                       : 'The Other Factory',
    'Kazookles'                         : 'The Other Factory',
}

df['Factory'] = df['Product Name'].map(product_factory_map)

# ── 5. Add Factory Coordinates ────────────────────────────────────────────────
factory_coords = {
    "Lot's O' Nuts"    : (32.881893, -111.768036),
    "Wicked Choccy's"  : (32.076176, -81.088371),
    'Sugar Shack'      : (48.11914,  -96.18115),
    'Secret Factory'   : (41.446333, -90.565487),
    'The Other Factory': (35.1175,   -89.971107),
}

df['Factory Lat'] = df['Factory'].map(lambda x: factory_coords[x][0])
df['Factory Lon'] = df['Factory'].map(lambda x: factory_coords[x][1])

# ── 6. Create Route Column ────────────────────────────────────────────────────
df['Route'] = df['Factory'] + ' → ' + df['State/Province']

# ── 7. Add Profit Margin ──────────────────────────────────────────────────────
df['Profit Margin (%)'] = (df['Gross Profit'] / df['Sales'] * 100).round(2)

# ── 8. Extract Time Features ──────────────────────────────────────────────────
df['Order Month'] = df['Order Date'].dt.month
df['Order Year']  = df['Order Date'].dt.year
df['Order Quarter'] = df['Order Date'].dt.quarter

# ── 9. Standardize Text Fields ────────────────────────────────────────────────
df['State/Province'] = df['State/Province'].str.strip().str.title()
df['City']           = df['City'].str.strip().str.title()
df['Region']         = df['Region'].str.strip().str.title()
df['Ship Mode']      = df['Ship Mode'].str.strip()

# ── 10. Save Cleaned Data ─────────────────────────────────────────────────────
df.to_csv('data/cleaned_data.csv', index=False)
print("Cleaned data saved! Final shape:", df.shape)
print("\nLead Time stats after fix:")
print(df['Lead Time (Days)'].describe())
print("\nSample routes:")
print(df['Route'].value_counts().head(10))
