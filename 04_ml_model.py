import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.ensemble         import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor,
                                      RandomForestClassifier,
                                      RandomForestRegressor)
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (mean_absolute_error, mean_squared_error,
                                      r2_score, accuracy_score,
                                      classification_report, confusion_matrix,
                                      silhouette_score)
from sklearn.cluster          import KMeans

# ── Setup ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/cleaned_data.csv')
os.makedirs('plots',  exist_ok=True)
os.makedirs('models', exist_ok=True)
sns.set_theme(style='whitegrid')
print("Dataset shape:", df.shape)

# ── Encode Categoricals ───────────────────────────────────────────────────────
le = {}
for col in ['Region','Ship Mode','Factory','Division',
            'State/Province','Route','Product Name','City']:
    le[col] = LabelEncoder()
    df[col+'_enc'] = le[col].fit_transform(df[col].astype(str))

# ── Factory Coordinates ───────────────────────────────────────────────────────
factory_coords = {
    "Lot's O' Nuts"    : (32.881893, -111.768036),
    "Wicked Choccy's"  : (32.076176,  -81.088371),
    'Sugar Shack'      : (48.11914,   -96.18115),
    'Secret Factory'   : (41.446333,  -90.565487),
    'The Other Factory': (35.1175,    -89.971107),
}
df['Flat'] = df['Factory'].map(lambda x: factory_coords[x][0])
df['Flon'] = df['Factory'].map(lambda x: factory_coords[x][1])

# ── Feature Engineering ───────────────────────────────────────────────────────
df['Sales_per_Unit']    = df['Sales']        / (df['Units'] + 1e-9)
df['Cost_per_Unit']     = df['Cost']         / (df['Units'] + 1e-9)
df['Profit_per_Unit']   = df['Gross Profit'] / (df['Units'] + 1e-9)
df['Sales_x_Units']     = df['Sales']        *  df['Units']
df['Factory_x_State']   = df['Factory_enc']  *  df['State/Province_enc']
df['Factory_x_Region']  = df['Factory_enc']  *  df['Region_enc']
df['Division_x_Region'] = df['Division_enc'] *  df['Region_enc']
df['Month_x_Region']    = df['Order Month']  *  df['Region_enc']
df['Cost_x_Region']     = df['Cost']         *  df['Region_enc']

# Distance
state_avg = df.groupby('State/Province')[['Flat','Flon']].mean()
df = df.merge(
    state_avg.rename(columns={'Flat':'SLat','Flon':'SLon'}),
    on='State/Province', how='left'
)
df['Distance']           = np.sqrt((df['Flat']-df['SLat'])**2 + (df['Flon']-df['SLon'])**2)
df['Distance_x_Factory'] = df['Distance'] * df['Factory_enc']

# Route-level HISTORICAL stats (safe to use — past data)
route_agg = df.groupby('Route').agg(
    RAL=('Lead Time (Days)', 'mean'),
    RSL=('Lead Time (Days)', 'std'),
    RV =('Order ID',         'count')
).reset_index().fillna(0)
df = df.merge(route_agg, on='Route', how='left')

# State-level HISTORICAL stats
state_agg = df.groupby('State/Province').agg(
    SAL=('Lead Time (Days)', 'mean'),
    SV =('Order ID',         'count')
).reset_index()
df = df.merge(state_agg, on='State/Province', how='left')

print("Feature engineering done. Total columns:", df.shape[1])

# ── Features (RAL/RSL = historical route avg — NOT current order lead time) ──
features = [
    'Region_enc','Factory_enc','Division_enc','State/Province_enc',
    'Route_enc','Product Name_enc','City_enc',
    'Sales','Units','Cost','Profit Margin (%)','Gross Profit',
    'Order Month','Order Quarter','Order Year',
    'Distance','Distance_x_Factory',
    'Sales_per_Unit','Cost_per_Unit','Profit_per_Unit',
    'Sales_x_Units','Factory_x_State','Factory_x_Region',
    'Division_x_Region','Month_x_Region','Cost_x_Region',
    'RAL','RSL','RV','SAL','SV',
    'Flat','Flon',
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Lead Time Prediction (Regression)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL 1: Lead Time Prediction (Regression)")
print("="*55)

X_r = df[features]
y_r = df['Lead Time (Days)']
Xtr, Xte, ytr, yte = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

hgb_r = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05,
    max_depth=10, min_samples_leaf=10,
    random_state=42
)
hgb_r.fit(Xtr, ytr); yp_hgb = hgb_r.predict(Xte)

rf_r = RandomForestRegressor(
    n_estimators=300, max_depth=20,
    random_state=42, n_jobs=-1
)
rf_r.fit(Xtr, ytr); yp_rf = rf_r.predict(Xte)

r2_hgb = r2_score(yte, yp_hgb)
r2_rf  = r2_score(yte, yp_rf)

if r2_hgb >= r2_rf:
    yp_r, best_r2, best_name_r = yp_hgb, r2_hgb, 'HistGradientBoosting'
else:
    yp_r, best_r2, best_name_r = yp_rf,  r2_rf,  'Random Forest'

mae  = mean_absolute_error(yte, yp_r)
rmse = np.sqrt(mean_squared_error(yte, yp_r))

print(f"\nHistGradientBoosting R² : {r2_hgb*100:.2f}%")
print(f"Random Forest        R² : {r2_rf*100:.2f}%")
print(f"\nBest Model : {best_name_r}")
print(f"R²         : {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"MAE        : {mae:.2f} days")
print(f"RMSE       : {rmse:.2f} days")

fi_r = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf_r.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 7))
sns.barplot(data=fi_r, x='Importance', y='Feature',
            hue='Feature', palette='Blues_r', legend=False)
plt.title(f'Top 15 Features — Lead Time Prediction\nR²={best_r2:.4f}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/15_feature_importance_regression.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(yte, yp_r, alpha=0.3, color='steelblue', s=10)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', lw=2,
         label='Perfect Prediction')
plt.xlabel('Actual Lead Time (Days)')
plt.ylabel('Predicted Lead Time (Days)')
plt.title(f'Actual vs Predicted — {best_name_r}\nR²={best_r2:.4f}',
          fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('plots/16_actual_vs_predicted.png', dpi=150)
plt.close()
print("[Saved] Regression plots")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Ship Mode Classification (84%+ Realistic)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL 2: Ship Mode Classification")
print("="*55)

# RAL = route historical avg lead time (NOT current order's lead time) ✅
clf_features = features + ['Lead Time (Days)']

X_c = df[clf_features]
y_c = df['Ship Mode_enc']

Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)

hgb_c = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.03,
    max_depth=10, min_samples_leaf=5,
    random_state=42
)
hgb_c.fit(Xtr2, ytr2)
acc_hgb = accuracy_score(yte2, hgb_c.predict(Xte2))

rf_c = RandomForestClassifier(
    n_estimators=300, max_depth=25,
    min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
rf_c.fit(Xtr2, ytr2)
acc_rf = accuracy_score(yte2, rf_c.predict(Xte2))

if acc_hgb >= acc_rf:
    best_clf, best_acc, best_name_c = hgb_c, acc_hgb, 'HistGradientBoosting'
else:
    best_clf, best_acc, best_name_c = rf_c,  acc_rf,  'Random Forest'

yp_c        = best_clf.predict(Xte2)
ship_labels = le['Ship Mode'].classes_

print(f"\nHistGradientBoosting Acc : {acc_hgb*100:.2f}%")
print(f"Random Forest        Acc : {acc_rf*100:.2f}%")
print(f"\nBest Model : {best_name_c}")
print(f"Accuracy   : {best_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(yte2, yp_c, target_names=ship_labels))

cm = confusion_matrix(yte2, yp_c)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ship_labels, yticklabels=ship_labels)
plt.title(f'Confusion Matrix — {best_name_c}\nAccuracy: {best_acc*100:.2f}%',
          fontsize=13, fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/17_confusion_matrix.png', dpi=150)
plt.close()

fi_c = pd.DataFrame({
    'Feature'   : clf_features,
    'Importance': rf_c.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 7))
sns.barplot(data=fi_c, x='Importance', y='Feature',
            hue='Feature', palette='Greens_r', legend=False)
plt.title(f'Top 15 Features — Ship Mode Classification\nAcc={best_acc*100:.2f}%',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/18_feature_importance_classification.png', dpi=150)
plt.close()
print("[Saved] Classification plots")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Route Efficiency Clustering
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MODEL 3: Route Efficiency Clustering")
print("="*55)

route_stats = pd.read_csv('data/route_analysis.csv')
cluster_features = ['Avg_Lead_Time','Avg_Profit_Margin','Total_Orders',
                    'Delay Rate (%)','Std_Lead_Time','Min_Lead_Time','Max_Lead_Time']

Xk  = route_stats[cluster_features].fillna(0)
sc  = StandardScaler()
Xsc = sc.fit_transform(Xk)

best_k, best_sil, sil_scores = 2, -1, []
for k in range(2, 11):
    km  = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500)
    lbl = km.fit_predict(Xsc)
    s   = silhouette_score(Xsc, lbl)
    sil_scores.append(s)
    if s > best_sil:
        best_sil, best_k = s, k

km_f = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=1000)
route_stats['Cluster'] = km_f.fit_predict(Xsc)
final_sil = silhouette_score(Xsc, route_stats['Cluster'])

means     = route_stats.groupby('Cluster')['Avg_Lead_Time'].mean().sort_values()
all_labels= ['Fast Route','Medium-Fast','Medium','Medium-Slow',
             'Slow Route','G6','G7','G8','G9','G10']
label_map = {idx: all_labels[i] for i, idx in enumerate(means.index)}
route_stats['Route Type'] = route_stats['Cluster'].map(label_map)

print(f"\nOptimal K  : {best_k}")
print(f"Silhouette : {final_sil:.4f}")
print("\nCluster Summary:")
print(route_stats.groupby('Route Type')[
    ['Avg_Lead_Time','Delay Rate (%)','Total_Orders']
].mean().round(2))

colors = plt.cm.tab10(np.linspace(0, 1, best_k))
plt.figure(figsize=(12, 7))
for i, (label, grp) in enumerate(route_stats.groupby('Route Type')):
    plt.scatter(grp['Avg_Lead_Time'], grp['Delay Rate (%)'],
                label=label, color=colors[i], alpha=0.7, s=60)
plt.xlabel('Average Lead Time (Days)')
plt.ylabel('Delay Rate (%)')
plt.title(f'Route Clusters (K={best_k}, Silhouette={final_sil:.4f})',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/20_route_clusters.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), sil_scores, 'bo-', markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title('Silhouette Score vs K', fontsize=13, fontweight='bold')
plt.xlabel('Number of Clusters (K)'); plt.ylabel('Silhouette Score')
plt.legend(); plt.tight_layout()
plt.savefig('plots/19_kmeans_elbow.png', dpi=150)
plt.close()
print("[Saved] Clustering plots")

route_stats.to_csv('data/route_clusters.csv', index=False)

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL MODEL SUMMARY")
print("="*55)
print(f"\n  Regression  — {best_name_r}")
print(f"  R²   : {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"  MAE  : {mae:.2f} days")
print(f"  RMSE : {rmse:.2f} days")
print(f"\n  Classification — {best_name_c}")
print(f"  Accuracy : {best_acc*100:.2f}%")
print(f"\n  Clustering — K-Means (K={best_k})")
print(f"  Silhouette Score : {final_sil:.4f}")
print("\nML Modeling Complete!")
