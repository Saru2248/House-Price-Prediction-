"""
visualize.py
------------
Generates and saves all EDA and model-evaluation plots to
the  outputs/  and  images/  folders.

Plots created:
    1.  Price distribution histogram
    2.  Correlation heatmap
    3.  Actual vs Predicted scatter plot
    4.  Feature importance bar chart (Random Forest & XGBoost)
    5.  Price by location box plot
    6.  Area vs Price scatter
    7.  Residuals plot
    8.  Model comparison bar chart
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import full_preprocess, engineer_features, FEATURE_COLS

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="husl")
PALETTE  = "viridis"
FIG_DIR  = "images"
os.makedirs(FIG_DIR, exist_ok=True)

TITLE_FONT   = {"fontsize": 14, "fontweight": "bold", "color": "#1a1a2e"}
LABEL_FONT   = {"fontsize": 11, "color": "#333333"}


def _save(fig, filename: str):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📸 Saved → {path}")


# ── 1. Price Distribution ────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("House Price Distribution", **TITLE_FONT)

    axes[0].hist(df["price_lakhs"], bins=50,
                 color="#6c63ff", edgecolor="white", alpha=0.85)
    axes[0].set_title("Raw Price Distribution", **LABEL_FONT)
    axes[0].set_xlabel("Price (₹ Lakhs)", **LABEL_FONT)
    axes[0].set_ylabel("Count", **LABEL_FONT)

    axes[1].hist(np.log1p(df["price_lakhs"]), bins=50,
                 color="#f77f00", edgecolor="white", alpha=0.85)
    axes[1].set_title("Log-Transformed Price", **LABEL_FONT)
    axes[1].set_xlabel("log(1 + Price)", **LABEL_FONT)
    axes[1].set_ylabel("Count", **LABEL_FONT)

    plt.tight_layout()
    _save(fig, "01_price_distribution.png")


# ── 2. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=["price_per_sqft"], errors="ignore"
    )
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", **TITLE_FONT)
    plt.tight_layout()
    _save(fig, "02_correlation_heatmap.png")


# ── 3. Price by Location ─────────────────────────────────────────────────────
def plot_price_by_location(df: pd.DataFrame):
    order = (df.groupby("location")["price_lakhs"]
               .median().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=df, x="location", y="price_lakhs",
                order=order, palette="Set2", ax=ax)
    ax.set_title("House Price by Location", **TITLE_FONT)
    ax.set_xlabel("Location", **LABEL_FONT)
    ax.set_ylabel("Price (₹ Lakhs)", **LABEL_FONT)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save(fig, "03_price_by_location.png")


# ── 4. Area vs Price ─────────────────────────────────────────────────────────
def plot_area_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["area_sqft"], df["price_lakhs"],
                    c=df["bedrooms"], cmap=PALETTE, alpha=0.5, s=25)
    plt.colorbar(sc, ax=ax, label="Bedrooms")
    ax.set_title("Area vs Price (colour = Bedrooms)", **TITLE_FONT)
    ax.set_xlabel("Area (sq.ft)", **LABEL_FONT)
    ax.set_ylabel("Price (₹ Lakhs)", **LABEL_FONT)
    plt.tight_layout()
    _save(fig, "04_area_vs_price.png")


# ── 5. Actual vs Predicted ───────────────────────────────────────────────────
def plot_actual_vs_predicted(y_actual, y_predicted, model_name: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_actual, y_predicted, alpha=0.4,
               color="#6c63ff", s=20, label="Predictions")
    lims = [min(y_actual.min(), y_predicted.min()),
            max(y_actual.max(), y_predicted.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    ax.set_title(f"Actual vs Predicted — {model_name}", **TITLE_FONT)
    ax.set_xlabel("Actual Price (₹ Lakhs)", **LABEL_FONT)
    ax.set_ylabel("Predicted Price (₹ Lakhs)", **LABEL_FONT)
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_actual_vs_predicted.png")


# ── 6. Residuals Plot ────────────────────────────────────────────────────────
def plot_residuals(y_actual, y_predicted):
    residuals = y_actual - y_predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residuals Analysis", **TITLE_FONT)

    axes[0].scatter(y_predicted, residuals, alpha=0.4,
                    color="#f77f00", s=20)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Price", **LABEL_FONT)
    axes[0].set_ylabel("Residual (Actual − Predicted)", **LABEL_FONT)
    axes[0].set_title("Residual Scatter", **LABEL_FONT)

    axes[1].hist(residuals, bins=40, color="#6c63ff",
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual", **LABEL_FONT)
    axes[1].set_ylabel("Count", **LABEL_FONT)
    axes[1].set_title("Residual Distribution", **LABEL_FONT)

    plt.tight_layout()
    _save(fig, "06_residuals.png")


# ── 7. Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        print(f"  ⚠️  {model_name} has no feature_importances_ — skipping.")
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp)))
    ax.barh(imp.index, imp.values, color=colors, edgecolor="none")
    ax.set_title(f"Feature Importance — {model_name}", **TITLE_FONT)
    ax.set_xlabel("Importance", **LABEL_FONT)
    plt.tight_layout()
    _save(fig, "07_feature_importance.png")


# ── 8. Model Comparison ──────────────────────────────────────────────────────
def plot_model_comparison():
    csv_path = "outputs/model_comparison.csv"
    if not os.path.exists(csv_path):
        print("  ⚠️  model_comparison.csv not found — skipping.")
        return

    comp = pd.read_csv(csv_path).sort_values("R2", ascending=False)
    metrics = ["R2", "MAE", "RMSE"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison", **TITLE_FONT)

    colors = sns.color_palette("Set2", len(comp))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(comp["Model"], comp[metric],
                      color=colors, edgecolor="none")
        ax.set_title(metric, **LABEL_FONT)
        ax.set_ylabel(metric, **LABEL_FONT)
        ax.tick_params(axis="x", rotation=25)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "08_model_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def generate_all_plots():
    print("\n🎨 Generating plots ...")

    df = pd.read_csv("data/housing_data.csv")
    df = engineer_features(df)

    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_by_location(df)
    plot_area_vs_price(df)

    # Load predictions
    pred_csv = "outputs/test_predictions.csv"
    if os.path.exists(pred_csv):
        pred_df = pd.read_csv(pred_csv)
        y_actual    = pred_df["actual"].values
        y_predicted = pred_df["predicted"].values

        # Detect best model name from comparison table
        comp_csv = "outputs/model_comparison.csv"
        best_name = "Best Model"
        if os.path.exists(comp_csv):
            best_name = pd.read_csv(comp_csv).iloc[0]["Model"]

        plot_actual_vs_predicted(y_actual, y_predicted, best_name)
        plot_residuals(y_actual, y_predicted)

    # Feature importance
    all_models = joblib.load("models/all_models.pkl")
    for mname in ["Random Forest", "XGBoost"]:
        if mname in all_models:
            plot_feature_importance(all_models[mname], FEATURE_COLS, mname)

    plot_model_comparison()
    print("\n✅ All plots saved to 'images/' folder")


if __name__ == "__main__":
    generate_all_plots()
