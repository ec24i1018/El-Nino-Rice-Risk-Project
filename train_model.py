import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD AND FILTER DATA
# ==========================================
print("Loading data...")
df = pd.read_csv("FINAL_GSC_DATASET_ALLSEASONS.csv")

# CRITICAL: Filter only for Kharif season
df = df[df['Season'] == 'Kharif'].copy()
print(f"Total Kharif records: {len(df)}")

# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================
# We drop 'State' and 'Season' because XGBoost can't read text (we handle this later if needed).
# We drop 'Rice_Yield' because we are predicting the RISK, not the exact yield.
# We drop 'Irrigation_Imputed' to prevent cheating.
features = [
    'Year', 
    'Season_Rainfall_mm', 
    'Season_Tmax_C', 
    'Season_Tmin_C', 
    'Season_Temp_C', 
    'Season_ONI', 
    'Irrigation_Pct'
]

X = df[features]
y = df['Target_Risk']

# ==========================================
# 3. TIME-SERIES SPLIT (NO RANDOM SHUFFLING)
# ==========================================
# We train on data before 2015, and test on 2015 onwards.
# This proves the model can predict the future!
train_mask = df['Year'] < 2015
test_mask = df['Year'] >= 2015

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training on {len(X_train)} records (Years < 2015)")
print(f"Testing on {len(X_test)} records (Years >= 2015)")

# ==========================================
# 4. TRAIN XGBOOST MODEL
# ==========================================
print("\nTraining XGBoost...")
# We use scale_pos_weight because you probably have more 0s (Normal) than 1s (Risk)
# This tells the model to pay extra attention to the rare 'Risk' events.
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train) 

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss',
    device='cpu'
)

model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE THE MODEL
# ==========================================
print("\nMaking predictions on the test set (2015-2019)...")
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Risk (1)']))

# ==========================================
# 6. FEATURE IMPORTANCE (For your report!)
# ==========================================
print("\nGenerating Feature Importance Chart...")
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain', max_num_features=10, height=0.5, title="What drives Rice Yield Risk?")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved chart as 'feature_importance.png'. Check your folder!")