# El Niño-Driven Rice Yield Risk Prediction 

## Problem Statement
El Niño events significantly disrupt the Indian monsoon, threatening food security. This project uses Machine Learning to predict the probability of rice yield declines (Risk) across Indian states based on climate and agricultural data.

## Dataset
- **Crop Yield & Irrigation:** Historical state-level data.
- **Climate Data:** NOAA ONI (El Niño Index), Rainfall, and Temperature.
- **Merged Data:** `FINAL_GSC_DATASET_ALLSEASONS.csv`

## Methodology
1. **EDA:** Analyzed ONI vs. Yield correlations.
2. **Feature Engineering:** Created a 5-year rolling average to define "Risk" (Target Variable).
3. **Modeling:** Compared Logistic Regression (Baseline), Random Forest, and XGBoost.
4. **Validation:** Used a strict Time-Series Split (Train: < 2015, Test: > 2015) to prevent data leakage.
5. **Metric:** Evaluated using the **F1-Score** due to class imbalance.

## Results
- **XGBoost** outperformed the other models.
- **Feature Importance:** Proved that the El Niño Index (ONI) is the primary driver of crop risk.
- **Historical Backtest:** The model successfully predicted the 2015 Super El Niño drought without being trained on it.
- **Future Projection:** Simulated a 2027 El Niño scenario, predicting >85% risk in states like Manipur and Jharkhand.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the model comparison: `python train_model.py`
3. Run historical validation: `python backtest.py`
4. Generate future charts: `python future_prediction.py`, `python future prediction graph.py`

## Part B: Budget Commander
**Learning Objective:** Teaches Descriptive Statistics and Sampling. Players must estimate a hidden population mean by purchasing random data samples.
**How to Play:** Spend limited gold to buy test battles. Analyze the histogram and standard deviation, then lock in your final guess to defeat the T-Rex.

## Limitations
The model assumes historical weather patterns remain constant for future projections. XGBoost is a black-box model, making exact causal inference difficult.
