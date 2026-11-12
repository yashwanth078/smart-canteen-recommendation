import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = Path(".")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
meal = pd.read_csv(DATA_PATH / "meal_info.csv")
center = pd.read_csv(DATA_PATH / "fulfilment_center_info.csv")

# Merge supporting tables
train = train.merge(meal, on="meal_id", how="left")
train = train.merge(center, on="center_id", how="left")

test = test.merge(meal, on="meal_id", how="left")
test = test.merge(center, on="center_id", how="left")

# Feature engineering
def prepare_features(df):
    df = df.copy()
    df["price_difference"] = df["base_price"] - df["checkout_price"]
    df["discount_percent"] = (
        df["price_difference"] / df["base_price"].replace(0, np.nan)
    ) * 100
    df["discount_percent"] = df["discount_percent"].fillna(0)

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    return df

train = prepare_features(train)
test = prepare_features(test)

target = "num_orders"

features = [
    "week",
    "center_id",
    "meal_id",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
    "category",
    "cuisine",
    "city_code",
    "region_code",
    "center_type",
    "op_area",
    "price_difference",
    "discount_percent",
    "week_sin",
    "week_cos"
]

features = [col for col in features if col in train.columns]

X = train[features]
y = np.log1p(train[target])

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ]
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

pred_log = pipeline.predict(X_valid)
actual = np.expm1(y_valid)
predicted = np.maximum(0, np.expm1(pred_log))

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("Model Evaluation")
print("-----------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.2f}")

# Train final model on all available data
pipeline.fit(X, y)

test_predictions = np.maximum(
    0,
    np.expm1(pipeline.predict(test[features]))
)

test["predicted_orders"] = np.round(test_predictions).astype(int)

# Recommended preparation quantity:
# 10% buffer protects against shortages while limiting overproduction
test["recommended_preparation"] = np.ceil(
    test["predicted_orders"] * 1.10
).astype(int)

# Risk classification
test["demand_risk"] = pd.cut(
    test["predicted_orders"],
    bins=[-1, 50, 150, np.inf],
    labels=["Low", "Medium", "High"]
)

# Export dashboard-ready data
test.to_csv("dashboard_predictions.csv", index=False)

# Create summary tables
weekly_summary = (
    train.groupby("week", as_index=False)["num_orders"]
    .sum()
    .rename(columns={"num_orders": "total_orders"})
)

meal_summary = (
    train.groupby(["meal_id", "category"], as_index=False)["num_orders"]
    .sum()
    .sort_values("num_orders", ascending=False)
)

center_summary = (
    train.groupby(["center_id", "city_code"], as_index=False)["num_orders"]
    .sum()
    .sort_values("num_orders", ascending=False)
)

weekly_summary.to_csv("weekly_summary.csv", index=False)
meal_summary.to_csv("meal_summary.csv", index=False)
center_summary.to_csv("center_summary.csv", index=False)

joblib.dump(pipeline, "demand_model.pkl")

print("\nFiles created:")
print("- dashboard_predictions.csv")
print("- weekly_summary.csv")
print("- meal_summary.csv")
print("- center_summary.csv")
print("- demand_model.pkl")
