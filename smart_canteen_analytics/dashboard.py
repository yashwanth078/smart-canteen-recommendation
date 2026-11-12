import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Smart Food Demand Dashboard",
    layout="wide"
)

st.title("Smart Food Demand and Waste-Reduction Dashboard")
st.caption(
    "Forecast-based meal preparation recommendations for food-service centers"
)

pred = pd.read_csv("dashboard_predictions.csv")
weekly = pd.read_csv("weekly_summary.csv")
meal = pd.read_csv("meal_summary.csv")
center = pd.read_csv("center_summary.csv")

# Sidebar filters
st.sidebar.header("Filters")

if "category" in pred.columns:
    categories = st.sidebar.multiselect(
        "Meal category",
        sorted(pred["category"].dropna().unique()),
        default=sorted(pred["category"].dropna().unique())
    )
    filtered = pred[pred["category"].isin(categories)]
else:
    filtered = pred.copy()

if "center_id" in filtered.columns:
    centers = st.sidebar.multiselect(
        "Fulfillment center",
        sorted(filtered["center_id"].unique()),
        default=sorted(filtered["center_id"].unique())
    )
    filtered = filtered[filtered["center_id"].isin(centers)]

# KPI calculations
total_predicted = int(filtered["predicted_orders"].sum())
total_recommended = int(filtered["recommended_preparation"].sum())
buffer_quantity = total_recommended - total_predicted
high_risk = int((filtered["demand_risk"] == "High").sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted orders", f"{total_predicted:,}")
col2.metric("Recommended preparation", f"{total_recommended:,}")
col3.metric("Safety-buffer quantity", f"{buffer_quantity:,}")
col4.metric("High-demand records", f"{high_risk:,}")

st.divider()

# Weekly demand chart
st.subheader("Historical weekly demand")

fig_week = px.line(
    weekly,
    x="week",
    y="total_orders",
    markers=True,
    labels={
        "week": "Week",
        "total_orders": "Total orders"
    }
)

st.plotly_chart(fig_week, use_container_width=True)

# Top meals
st.subheader("Top meal categories by demand")

top_meals = (
    filtered.groupby("category", as_index=False)["predicted_orders"]
    .sum()
    .sort_values("predicted_orders", ascending=False)
    .head(10)
)

fig_meals = px.bar(
    top_meals,
    x="category",
    y="predicted_orders",
    color="predicted_orders",
    labels={
        "category": "Meal category",
        "predicted_orders": "Predicted orders"
    }
)

st.plotly_chart(fig_meals, use_container_width=True)

# Center performance
st.subheader("Center-level demand")

center_view = (
    filtered.groupby("center_id", as_index=False)["predicted_orders"]
    .sum()
    .sort_values("predicted_orders", ascending=False)
)

fig_centers = px.bar(
    center_view,
    x="center_id",
    y="predicted_orders",
    color="predicted_orders",
    labels={
        "center_id": "Center",
        "predicted_orders": "Predicted orders"
    }
)

st.plotly_chart(fig_centers, use_container_width=True)

# Demand risk
st.subheader("Demand-risk distribution")

risk_view = (
    filtered["demand_risk"]
    .value_counts()
    .rename_axis("demand_risk")
    .reset_index(name="records")
)

fig_risk = px.pie(
    risk_view,
    names="demand_risk",
    values="records",
    hole=0.4
)

st.plotly_chart(fig_risk, use_container_width=True)

# Recommendations table
st.subheader("Preparation recommendations")

columns = [
    col for col in [
        "week",
        "center_id",
        "meal_id",
        "category",
        "cuisine",
        "predicted_orders",
        "recommended_preparation",
        "demand_risk"
    ]
    if col in filtered.columns
]

st.dataframe(
    filtered[columns].sort_values(
        "predicted_orders",
        ascending=False
    ).head(100),
    use_container_width=True
)

st.download_button(
    label="Download filtered recommendations",
    data=filtered.to_csv(index=False),
    file_name="filtered_preparation_recommendations.csv",
    mime="text/csv"
)
