import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD AND TRAIN MODEL
# ==========================================
print("Loading data and training historical model...")
df = pd.read_csv("FINAL_GSC_DATASET_ALLSEASONS.csv")
df = df[df['Season'] == 'Kharif'].copy()

features = ['Year', 'Season_Rainfall_mm', 'Season_Tmax_C', 'Season_Tmin_C', 
            'Season_Temp_C', 'Season_ONI', 'Irrigation_Pct']

# Train on 1998-2010 data
train_mask = df['Year'] <= 2010
train_df = df[train_mask]
X_train, y_train = train_df[features], train_df['Target_Risk']

scale_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1)
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, 
                          scale_pos_weight=scale_weight, random_state=42, device='cpu')
model.fit(X_train, y_train)

# ==========================================
# 2. PREPARE DATA AND GENERATE PREDICTIONS
# ==========================================
# Use historical average weather as baseline
baseline = df.groupby('State')[['Season_Rainfall_mm', 'Season_Tmax_C', 'Season_Tmin_C', 'Season_Temp_C', 'Irrigation_Pct']].mean().reset_index()

historical_oni = {
    2011: -0.6, 2012: 0.3, 2013: -0.3, 2014: 0.1, 2015: 1.8, 2016: -0.4, 2017: 0.1, 
    2018: 0.2, 2019: 0.3, 2020: -0.6, 2021: -0.4, 2022: -0.9, 2023: 1.3, 2024: 0.1, 
    2025: 0.0, 2026: -0.4
}

results_list = []

# Simulate each year to build results_df
for year, oni_val in historical_oni.items():
    sim_df = baseline.copy()
    sim_df['Year'] = year
    sim_df['Season_ONI'] = oni_val
    
    probs = model.predict_proba(sim_df[features])[:, 1]
    avg_risk = np.mean(probs) * 100
    
    results_list.append({
        'Year': year,
        'ONI_Value': oni_val,
        'Predicted_National_Risk_%': avg_risk
    })

# THIS is the variable that was missing!
results_df = pd.DataFrame(results_list)

# ==========================================
# 3. CREATE THE VISUALIZATION (WITH PROPER LEGEND)
# ==========================================
print("\nGenerating historical chart...")
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot the Predicted Risk on the primary axis (Left)
line1 = sns.lineplot(
    data=results_df, x='Year', y='Predicted_National_Risk_%', 
    marker='o', color='blue', linewidth=2.5, label='Model-Predicted Rice Risk (%)', ax=ax1
)
ax1.set_ylabel('Predicted Risk Probability (%)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add the danger threshold line to the first axis
line2 = ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='High Risk Threshold (50%)')

# Create a secondary axis for the ONI (Right)
ax2 = ax1.twinx()
line3 = sns.lineplot(
    data=results_df, x='Year', y='ONI_Value', 
    marker='s', color='red', alpha=0.6, linewidth=2, label='ONI Value (El Niño / La Niña)', ax=ax2
)
ax2.set_ylabel('ONI Value (Oceanic Niño Index)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine the legends properly
lines = [line1.lines[0], line3.lines[0], line2]
labels = ['Model-Predicted Rice Risk (%)', 'ONI Value (El Niño / La Niña)', 'High Risk Threshold (50%)']

# Place the combined legend
ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, shadow=True)

# Formatting
plt.title('Historical Validation: Model-Predicted Rice Risk vs El Niño (2011-2026)', fontsize=16, pad=15)
ax1.set_xlabel('Year', fontsize=12)

# Force X-axis to show whole years
ax1.set_xticks(range(2011, 2027, 1)) 

plt.tight_layout()
plt.savefig("Historical_Validation.png", dpi=300)
print("Success! Chart saved as 'Historical_Validation.png'")
plt.show()