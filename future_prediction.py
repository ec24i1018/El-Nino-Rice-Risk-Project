import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD AND TRAIN THE MODEL
# ==========================================
print("Training model...")
df = pd.read_csv("FINAL_GSC_DATASET_ALLSEASONS.csv")
df = df[df['Season'] == 'Kharif'].copy()

features = [
    'Year', 'Season_Rainfall_mm', 'Season_Tmax_C', 'Season_Tmin_C', 
    'Season_Temp_C', 'Season_ONI', 'Irrigation_Pct'
]
X = df[features]
y = df['Target_Risk']

scale_weight = (len(y) - sum(y)) / sum(y) 
model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    scale_pos_weight=scale_weight, random_state=42, device='cpu'
)
model.fit(X, y)

# ==========================================
# 2. SET UP 2027 SCENARIO (Strong El Niño)
# ==========================================
# We use 2019 weather data as the baseline, change year to 2027
future_df = df[df['Year'] == 2019].copy()
future_df['Year'] = 2027 

# Simulate a Strong El Niño for 2027
future_df['Season_ONI'] = 1.5 

# Predict probabilities
risk_probs = model.predict_proba(future_df[features])[:, 1]
future_df['Risk_Probability'] = risk_probs * 100 # Convert to percentage

# Sort states by highest risk
results = future_df[['State', 'Risk_Probability']].sort_values(by='Risk_Probability', ascending=False)

# ==========================================
# 3. CREATE THE VISUALIZATION
# ==========================================
print("Generating chart...")
plt.figure(figsize=(12, 8))

# Use seaborn to make a color-coded horizontal bar chart
# Red = High Risk, Green = Low Risk
sns.barplot(
    x='Risk_Probability', 
    y='State', 
    data=results, 
    palette='RdYlGn', # Red-Yellow-Green reversed (Red is high)
    hue='State',
    legend=False
)

# Add titles and labels
plt.title('Projected Rice Yield Risk in 2027 (Strong El Niño Scenario, ONI = +1.5)', fontsize=16, pad=20)
plt.xlabel('Probability of Yield Decline (%)', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.xlim(0, 100) # Scale from 0 to 100%

# Add a vertical line at 50% to show the "danger threshold"
plt.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='High Risk Threshold (50%)')
plt.legend()

# Add exact percentage text to the end of each bar
for index, value in enumerate(results['Risk_Probability']):
    plt.text(value + 1, index, f'{value:.1f}%', va='center', fontsize=10)

plt.tight_layout()

# Save the image and show it
plt.savefig("Rice_Risk_2027.png", dpi=300)
print("Success! Chart saved as 'Rice_Risk_2027.png' in your folder.")
plt.show()