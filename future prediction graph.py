import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. TRAIN THE MODEL
# ==========================================
print("Loading and training model...")
df = pd.read_csv("FINAL_GSC_DATASET_ALLSEASONS.csv")
df = df[df['Season'] == 'Kharif'].copy()

features = ['Year', 'Season_Rainfall_mm', 'Season_Tmax_C', 'Season_Tmin_C', 
            'Season_Temp_C', 'Season_ONI', 'Irrigation_Pct']

# Train on ALL data up to 2019
X_train, y_train = df[features], df['Target_Risk']
scale_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1)
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, 
                          scale_pos_weight=scale_weight, random_state=42, device='cpu')
model.fit(X_train, y_train)

# Define the baseline (average historical weather/irrigation per state)
baseline = df.groupby('State')[['Season_Rainfall_mm', 'Season_Tmax_C', 'Season_Tmin_C', 'Season_Temp_C', 'Irrigation_Pct']].mean().reset_index()

# ==========================================
# 2. SET UP THE TIMELINE (Past + Future)
# ==========================================
# Historical ONI data (2012 to 2025)
past_oni = {2012: 0.3, 2013: -0.3, 2014: 0.1, 2015: 1.8, 
            2016: -0.4, 2017: 0.1, 2018: 0.2, 2019: 0.3, 2020: -0.6, 2021: -0.4, 
            2022: -0.9, 2023: 1.3, 2024: 0.1, 2025: -0.4}

# Future scenarios (2026 to 2030)
future_oni = {2026: 0.5, 2027: 1.8, 2028: 1.2, 2029: 0.2, 2030: -0.5}

# Combine them
all_years = list(past_oni.keys()) + list(future_oni.keys())
all_oni = list(past_oni.values()) + list(future_oni.values())

results = []

# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================
for year, oni_val in zip(all_years, all_oni):
    sim_df = baseline.copy()
    sim_df['Year'] = year
    sim_df['Season_ONI'] = oni_val
    
    probs = model.predict_proba(sim_df[features])[:, 1]
    avg_risk = np.mean(probs) * 100 
    
    results.append({'Year': year, 'ONI': oni_val, 'Risk': avg_risk})

plot_df = pd.DataFrame(results)

# Split into past and future for coloring
past_df = plot_df[plot_df['Year'] <= 2025]
future_df = plot_df[plot_df['Year'] >= 2025] # Overlap at 2025 for a continuous line

# ==========================================
# 4. CREATE THE VISUALIZATION
# ==========================================
print("Generating the presentation chart...")
plt.figure(figsize=(14, 6))

# Plot Past (Blue)
sns.lineplot(data=past_df, x='Year', y='Risk', marker='o', color='blue', linewidth=2.5, label='Historical Predicted Risk')

# Plot Future (Dashed Red)
sns.lineplot(data=future_df, x='Year', y='Risk', marker='o', color='red', linewidth=2.5, linestyle='--', label='Future Projected Risk')

# Add the "Present Day" divider line at 2025
plt.axvline(x=2025, color='gray', linestyle=':', linewidth=2)
plt.text(2025.1, 40, 'Present Day (2026)', color='gray', fontsize=12, fontweight='bold')

# Add the high-risk threshold
plt.axhline(y=50, color='black', linestyle='--', alpha=0.3, label='High Risk Threshold (50%)')

# Formatting
plt.title('Rice Yield Risk Projection: Historical Validation vs. Future Scenarios (2012-2030)', fontsize=16, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Predicted National Risk Probability (%)', fontsize=12)
plt.ylim(35, 75) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Force the X-axis to only show whole years
plt.xticks(range(2012, 2031, 2)) 

# Highlight the 2027 peak
peak_year = 2027
peak_risk = future_df[future_df['Year'] == 2027]['Risk'].values[0]
plt.annotate(f'Projected Crisis\n(El Niño +1.8)\nRisk: {peak_risk:.1f}%', 
             xy=(peak_year, peak_risk), xytext=(peak_year-2, peak_risk+5),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=11, color='red', fontweight='bold')

plt.legend(loc='upper left')
plt.tight_layout()

# Save and show
plt.savefig("Future_Projection_Plot_2026.png", dpi=300)
print("Success! Chart saved as 'Future_Projection_Plot_2026.png'")
plt.show()