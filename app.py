import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Shipping Efficiency",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df         = pd.read_csv('data/cleaned_data.csv', parse_dates=['Order Date', 'Ship Date'])
    route_df   = pd.read_csv('data/route_analysis.csv')
    state_df   = pd.read_csv('data/state_analysis.csv')
    cluster_df = pd.read_csv('data/route_clusters.csv')
    return df, route_df, state_df, cluster_df

df, route_df, state_df, cluster_df = load_data()

# ── US State Abbreviations (for map) ─────────────────────────────────────────
state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
    'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
    'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC',
    'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
    'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN',
    'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3082/3082048.png", width=80)
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.markdown("**Shipping Route Efficiency**")
st.sidebar.divider()

# Date Range
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()
date_range = st.sidebar.date_input(
    "📅 Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region Filter
regions    = ['All'] + sorted(df['Region'].dropna().unique().tolist())
sel_region = st.sidebar.selectbox("🗺️ Region", regions)

# State Filter
states    = ['All'] + sorted(df['State/Province'].dropna().unique().tolist())
sel_state = st.sidebar.selectbox("📍 State", states)

# Ship Mode Filter
modes    = ['All'] + sorted(df['Ship Mode'].dropna().unique().tolist())
sel_mode = st.sidebar.multiselect("🚚 Ship Mode", modes[1:], default=modes[1:])

# Lead Time Threshold
threshold = st.sidebar.slider(
    "⏱️ Delay Threshold (Days)",
    min_value=int(df['Lead Time (Days)'].min()),
    max_value=int(df['Lead Time (Days)'].max()),
    value=int(df['Lead Time (Days)'].mean() + df['Lead Time (Days)'].std())
)

st.sidebar.divider()
st.sidebar.caption("Factory-to-Customer Shipping\nRoute Efficiency Analysis")

# ── Apply Filters ─────────────────────────────────────────────────────────────
filtered = df.copy()

if len(date_range) == 2:
    filtered = filtered[
        (filtered['Order Date'].dt.date >= date_range[0]) &
        (filtered['Order Date'].dt.date <= date_range[1])
    ]

if sel_region != 'All':
    filtered = filtered[filtered['Region'] == sel_region]

if sel_state != 'All':
    filtered = filtered[filtered['State/Province'] == sel_state]

if sel_mode:
    filtered = filtered[filtered['Ship Mode'].isin(sel_mode)]

filtered['Is_Delayed'] = filtered['Lead Time (Days)'] > threshold

# ── Header KPIs ───────────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy — Shipping Route Efficiency Dashboard")
st.caption(f"Showing **{len(filtered):,}** orders after filters")
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Total Orders",       f"{len(filtered):,}")
k2.metric("⏱️ Avg Lead Time",      f"{filtered['Lead Time (Days)'].mean():.0f} days")
k3.metric("⚠️ Delayed Orders",     f"{filtered['Is_Delayed'].sum():,}",
          delta=f"{filtered['Is_Delayed'].mean()*100:.1f}% delay rate",
          delta_color="inverse")
k4.metric("💰 Total Sales",        f"${filtered['Sales'].sum():,.0f}")
k5.metric("📈 Avg Profit Margin",  f"{filtered['Profit Margin (%)'].mean():.1f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Route Efficiency",
    "🗺️  Geographic Map",
    "🚚 Ship Mode Analysis",
    "🔍 Route Drill-Down"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Route Efficiency Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Route Efficiency Overview")

    # Recompute route stats from filtered data
    f_route = filtered.groupby('Route').agg(
        Avg_Lead_Time     = ('Lead Time (Days)', 'mean'),
        Total_Orders      = ('Order ID',         'count'),
        Delay_Rate        = ('Is_Delayed',        'mean'),
        Avg_Profit_Margin = ('Profit Margin (%)', 'mean'),
    ).reset_index()
    f_route['Delay Rate (%)']  = (f_route['Delay_Rate'] * 100).round(1)
    f_route['Avg_Lead_Time']   = f_route['Avg_Lead_Time'].round(1)
    f_route['Efficiency Score']= (
        100 * (f_route['Avg_Lead_Time'].max() - f_route['Avg_Lead_Time']) /
        (f_route['Avg_Lead_Time'].max() - f_route['Avg_Lead_Time'].min() + 1e-9)
    ).round(1)

    reliable = f_route[f_route['Total_Orders'] >= 5]
    top10    = reliable.nlargest(10, 'Efficiency Score')
    bot10    = reliable.nsmallest(10, 'Efficiency Score')

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(top10, x='Efficiency Score', y='Route', orientation='h',
                     color='Efficiency Score', color_continuous_scale='Greens',
                     title='🏆 Top 10 Most Efficient Routes',
                     labels={'Efficiency Score': 'Score (0-100)'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(bot10, x='Efficiency Score', y='Route', orientation='h',
                     color='Efficiency Score', color_continuous_scale='Reds_r',
                     title='⚠️ Bottom 10 Least Efficient Routes',
                     labels={'Efficiency Score': 'Score (0-100)'})
        fig.update_layout(yaxis={'categoryorder': 'total descending'}, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Leaderboard Table
    st.subheader("📋 Route Performance Leaderboard")
    leaderboard = reliable.sort_values('Efficiency Score', ascending=False).reset_index(drop=True)
    leaderboard.index += 1
    leaderboard['Avg_Lead_Time']       = leaderboard['Avg_Lead_Time'].map('{:.1f} days'.format)
    leaderboard['Avg_Profit_Margin']   = leaderboard['Avg_Profit_Margin'].map('{:.1f}%'.format)
    leaderboard['Delay Rate (%)']      = leaderboard['Delay Rate (%)'].map('{:.1f}%'.format)
    leaderboard['Efficiency Score']    = leaderboard['Efficiency Score'].map('{:.1f}'.format)
    st.dataframe(
        leaderboard[['Route', 'Total_Orders', 'Avg_Lead_Time',
                     'Delay Rate (%)', 'Avg_Profit_Margin', 'Efficiency Score']],
        use_container_width=True, height=350
    )

    # Lead Time Distribution
    st.subheader("📉 Lead Time Distribution")
    fig = px.histogram(filtered, x='Lead Time (Days)', nbins=40,
                       color='Ship Mode', barmode='overlay',
                       title='Lead Time Distribution by Ship Mode',
                       opacity=0.7)
    fig.add_vline(x=threshold, line_dash='dash', line_color='red',
                  annotation_text=f'Delay Threshold ({threshold}d)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geographic Map
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Geographic Shipping Efficiency Map")

    # State-level stats from filtered
    f_state = filtered.groupby('State/Province').agg(
        Total_Orders  = ('Order ID',          'count'),
        Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
        Delay_Rate    = ('Is_Delayed',          'mean'),
        Total_Sales   = ('Sales',              'sum'),
    ).reset_index()
    f_state['Delay_Rate (%)'] = (f_state['Delay_Rate'] * 100).round(1)
    f_state['Avg_Lead_Time']  = f_state['Avg_Lead_Time'].round(1)
    f_state['State_Code']     = f_state['State/Province'].map(state_abbrev)
    f_state = f_state.dropna(subset=['State_Code'])

    map_metric = st.radio(
        "Color map by:",
        ['Avg_Lead_Time', 'Delay_Rate (%)', 'Total_Orders'],
        horizontal=True,
        format_func=lambda x: {
            'Avg_Lead_Time': '⏱ Avg Lead Time',
            'Delay_Rate (%)': '⚠️ Delay Rate %',
            'Total_Orders': '📦 Order Volume'
        }[x]
    )

    fig = px.choropleth(
        f_state,
        locations='State_Code',
        locationmode='USA-states',
        color=map_metric,
        scope='usa',
        color_continuous_scale='YlOrRd',
        hover_name='State/Province',
        hover_data={
            'Total_Orders': True,
            'Avg_Lead_Time': True,
            'Delay_Rate (%)': True,
            'State_Code': False
        },
        title=f'US Shipping Efficiency Heatmap — {map_metric.replace("_", " ")}'
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    # Factory Locations
    st.subheader("🏭 Factory Locations")
    factory_info = pd.DataFrame({
        'Factory'  : ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack",
                      "Secret Factory", "The Other Factory"],
        'Latitude' : [32.881893, 32.076176, 48.11914,  41.446333, 35.1175],
        'Longitude': [-111.768036, -81.088371, -96.18115, -90.565487, -89.971107],
        'State'    : ['Arizona', 'Georgia', 'Minnesota', 'Illinois', 'Tennessee']
    })

    fig2 = px.scatter_geo(
        factory_info,
        lat='Latitude', lon='Longitude',
        text='Factory',
        hover_name='Factory',
        hover_data={'State': True, 'Latitude': False, 'Longitude': False},
        scope='usa',
        title='Factory Locations Across the US',
        size_max=20
    )
    fig2.update_traces(marker=dict(size=14, color='royalblue', symbol='star'))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Bottleneck Table
    st.subheader("🚨 Geographic Bottlenecks")
    bottleneck = f_state.sort_values('Delay_Rate (%)', ascending=False).head(10)
    st.dataframe(
        bottleneck[['State/Province', 'Total_Orders', 'Avg_Lead_Time', 'Delay_Rate (%)']],
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ship Mode Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ship Mode Performance Analysis")

    shipmode = filtered.groupby('Ship Mode').agg(
        Total_Orders  = ('Order ID',          'count'),
        Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
        Avg_Sales     = ('Sales',             'mean'),
        Avg_Margin    = ('Profit Margin (%)', 'mean'),
        Delay_Rate    = ('Is_Delayed',         'mean'),
    ).reset_index()
    shipmode['Delay Rate (%)'] = (shipmode['Delay_Rate'] * 100).round(1)
    shipmode = shipmode.round(2)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.bar(shipmode, x='Ship Mode', y='Avg_Lead_Time',
                     color='Ship Mode', title='⏱ Avg Lead Time by Ship Mode',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(shipmode, x='Ship Mode', y='Delay Rate (%)',
                     color='Ship Mode', title='⚠️ Delay Rate by Ship Mode',
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.bar(shipmode, x='Ship Mode', y='Avg_Margin',
                     color='Ship Mode', title='💰 Avg Profit Margin by Ship Mode',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap Region vs Ship Mode
    st.subheader("🔥 Lead Time Heatmap: Region vs Ship Mode")
    pivot = filtered.pivot_table(
        values='Lead Time (Days)', index='Region',
        columns='Ship Mode', aggfunc='mean'
    ).round(1)

    fig = px.imshow(pivot, text_auto=True, color_continuous_scale='YlOrRd',
                    title='Average Lead Time (Days): Region × Ship Mode',
                    aspect='auto')
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Box Plot
    st.subheader("📊 Lead Time Spread by Ship Mode")
    fig = px.box(filtered, x='Ship Mode', y='Lead Time (Days)',
                 color='Ship Mode',
                 title='Lead Time Distribution per Ship Mode',
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.add_hline(y=threshold, line_dash='dash', line_color='red',
                  annotation_text='Delay Threshold')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Ship Mode Summary Table
    st.subheader("📋 Ship Mode Summary Table")
    st.dataframe(
        shipmode[['Ship Mode', 'Total_Orders', 'Avg_Lead_Time',
                  'Delay Rate (%)', 'Avg_Sales', 'Avg_Margin']],
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Route Drill-Down
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Route Drill-Down")

    c1, c2 = st.columns(2)
    with c1:
        sel_factory = st.selectbox("🏭 Select Factory",
                                   ['All'] + sorted(filtered['Factory'].dropna().unique()))
    with c2:
        sel_drillstate = st.selectbox("📍 Select State",
                                      ['All'] + sorted(filtered['State/Province'].dropna().unique()))

    drilled = filtered.copy()
    if sel_factory != 'All':
        drilled = drilled[drilled['Factory'] == sel_factory]
    if sel_drillstate != 'All':
        drilled = drilled[drilled['State/Province'] == sel_drillstate]

    st.caption(f"Showing **{len(drilled):,}** orders")

    # State-Level Performance
    st.subheader("📍 State-Level Performance")
    state_perf = drilled.groupby('State/Province').agg(
        Orders        = ('Order ID',          'count'),
        Avg_Lead_Time = ('Lead Time (Days)',   'mean'),
        Delay_Rate    = ('Is_Delayed',          'mean'),
        Total_Sales   = ('Sales',              'sum'),
    ).reset_index().sort_values('Avg_Lead_Time')
    state_perf['Delay Rate (%)'] = (state_perf['Delay_Rate'] * 100).round(1)
    state_perf['Avg_Lead_Time']  = state_perf['Avg_Lead_Time'].round(1)

    fig = px.bar(state_perf, x='State/Province', y='Avg_Lead_Time',
                 color='Delay Rate (%)', color_continuous_scale='RdYlGn_r',
                 title='Avg Lead Time by State (colored by Delay Rate)',
                 labels={'Avg_Lead_Time': 'Avg Lead Time (Days)'})
    fig.update_layout(height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Order Timeline
    st.subheader("📅 Order Shipment Timeline")
    timeline = drilled.groupby('Order Date').agg(
        Orders        = ('Order ID',        'count'),
        Avg_Lead_Time = ('Lead Time (Days)', 'mean'),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=timeline['Order Date'], y=timeline['Orders'],
                         name='Orders', marker_color='steelblue', opacity=0.6),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=timeline['Order Date'], y=timeline['Avg_Lead_Time'],
                             name='Avg Lead Time', line=dict(color='red', width=2)),
                  secondary_y=True)
    fig.update_layout(title='Daily Orders & Avg Lead Time Over Time', height=400)
    fig.update_yaxes(title_text='Orders',           secondary_y=False)
    fig.update_yaxes(title_text='Avg Lead Time (Days)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Route Clusters
    st.subheader("🔵 Route Efficiency Clusters")
    fig = px.scatter(cluster_df, x='Avg_Lead_Time', y='Delay Rate (%)',
                     color='Route Type', size='Total_Orders',
                     hover_name='Route',
                     color_discrete_map={
                         'Fast Route':   'green',
                         'Medium Route': 'orange',
                         'Slow Route':   'red'
                     },
                     title='Route Clusters: Fast vs Medium vs Slow',
                     labels={'Avg_Lead_Time': 'Avg Lead Time (Days)'})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data View
    st.subheader("📄 Raw Order Data")
    show_cols = ['Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                 'Factory', 'State/Province', 'Region', 'Product Name',
                 'Lead Time (Days)', 'Sales', 'Profit Margin (%)', 'Is_Delayed']
    st.dataframe(drilled[show_cols].reset_index(drop=True), use_container_width=True, height=350)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Nassau Candy Distributor — Factory-to-Customer Shipping Route Efficiency Analysis")
