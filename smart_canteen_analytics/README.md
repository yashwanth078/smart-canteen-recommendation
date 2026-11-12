# Smart Canteen Food-Waste Analytics

## Problem Statement

Food-service organizations frequently overprepare meals because they lack an analytical method for estimating demand for each meal and location. This project uses a meal-demand dataset to analyze weekly order patterns, meal categories, fulfillment-center characteristics, and promotional factors. It builds an executable demand-prediction model and an interactive dashboard that identifies high-demand meals, detects possible overstocking, and recommends preparation quantities to reduce avoidable food waste.

## Dataset

Place the following CSV files in the same folder as this project:

- `train.csv`
- `test.csv`
- `meal_info.csv`
- `fulfilment_center_info.csv`

You can download them from the Kaggle "Food Demand Forecasting" dataset:
https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting

A small synthetic sample dataset is included so you can test the code immediately. Replace it with the full Kaggle data for better results.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the analysis script:

```bash
python analysis.py
```

This will:

- Clean and merge data
- Engineer features
- Train a Random Forest model
- Evaluate performance (MAE, RMSE, R2)
- Generate predictions and recommendations
- Save `dashboard_predictions.csv`, `weekly_summary.csv`, `meal_summary.csv`, `center_summary.csv`, and `demand_model.pkl`

2. Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

This opens an interactive dashboard with:

- Predicted orders and recommended preparation quantities
- Weekly demand trends
- Top meal categories
- Center-level demand
- Demand-risk distribution
- Downloadable recommendations table

## Files

- `analysis.py` – Data processing, modeling, and prediction
- `dashboard.py` – Streamlit dashboard
- `requirements.txt` – Python dependencies
- `README.md` – This file
- `train.csv`, `test.csv`, `meal_info.csv`, `fulfilment_center_info.csv` – Dataset files

## Notes

- The model predicts `num_orders` for each center–meal–week combination.
- Recommended preparation = predicted orders × 1.10 (10% safety buffer).
- You can adjust the buffer factor in `analysis.py` if needed.
