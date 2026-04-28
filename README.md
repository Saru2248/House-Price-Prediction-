# House Price Prediction 🏠

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A complete, industry-grade Machine Learning project** that predicts residential property prices using Linear Regression, Decision Tree, Random Forest, and XGBoost models — built with synthetic Indian real-estate data.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Industry Relevance](#-industry-relevance)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Models Used](#-models-used)
- [Results](#-results)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Screenshots](#-screenshots)
- [Key Learnings](#-key-learnings)
- [Interview Q&A](#-interview-qa)


---

## 🏠 Project Overview

House Price Prediction is one of the most impactful real-world applications of supervised machine learning. This project simulates a full, production-grade pipeline that a Data Scientist would build at a real estate company like MagicBricks, 99acres, or a bank's loan valuation team.

The pipeline covers:
- **Synthetic dataset generation** (2,000 property records)
- **Data cleaning** (outlier removal, missing value imputation)
- **Feature engineering** (floor ratio, newness score, amenity score)
- **Model training & comparison** (4 regression algorithms)
- **Model evaluation** (MAE, RMSE, R², MAPE, 5-fold CV)
- **Interactive price prediction** (CLI)
- **Rich visualisations** (8 publication-quality charts)

---

## ❓ Problem Statement

**Challenge:** Property buyers, sellers, banks, and real-estate portals lack a data-driven, objective method to determine fair property prices.

**Solution:** Build a regression model trained on historical property features (location, area, amenities, age, etc.) that can predict the market price of any residential property.

**Business Impact:**
| Stakeholder | Benefit |
|---|---|
| 🏦 Bank / NBFCs | Accurate loan-to-value assessment |
| 🛒 Buyer | Avoid overpaying; negotiate with data |
| 🏡 Seller | Price property competitively |
| 🏢 Real Estate Portal | Power automated valuation tools |
| 📊 Investor | Identify undervalued properties |

---

## 🏭 Industry Relevance

This type of system is used in production at:
- **MagicBricks / 99acres / Housing.com** — AVM (Automated Valuation Models)
- **HDFC / SBI / ICICI Bank** — Property collateral valuation
- **PropTech startups** — AI-powered price estimates
- **Insurance companies** — Property underwriting
- **Government** — Stamp duty guidance, circle rates

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  area_sqft | bedrooms | bathrooms | location | age | etc.   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                          │
│  ① Missing value imputation   (median / mode)               │
│  ② Outlier removal            (IQR method)                  │
│  ③ Feature Engineering        (floor_ratio, newness_score)  │
│  ④ Categorical Encoding       (LabelEncoder)                │
│  ⑤ Feature Scaling            (StandardScaler)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODEL LAYER                                │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ Linear Regression│  │   Decision Tree Regressor        │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Random Forest   │  │   XGBoost Regressor              │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                               │
│         🏡  Predicted Price: ₹ 85.47 Lakhs                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Boosting | XGBoost |
| Model Persistence | Joblib |
| Environment | Virtual Environment (venv) |

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Synthetically generated (mimics Indian real-estate) |
| Records | 2,000 properties |
| Features | 11 input features + 1 target |
| Target | `price_lakhs` (property price in ₹ Lakhs) |

**Features:**

| Feature | Type | Description |
|---|---|---|
| `area_sqft` | Numeric | Built-up area in sq. ft. |
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms |
| `balconies` | Numeric | Number of balconies |
| `location` | Categorical | Locality name |
| `age_years` | Numeric | Age of property in years |
| `parking` | Numeric | Parking spots (0–2) |
| `furnishing` | Categorical | Unfurnished / Semi / Furnished |
| `floor` | Numeric | Floor number |
| `total_floors` | Numeric | Total floors in building |
| `distance_city_km` | Numeric | Distance from city centre |

**Engineered Features:**

| Feature | Formula |
|---|---|
| `floor_ratio` | `floor / total_floors` |
| `newness_score` | `1 − age_years/30` |
| `amenity_score` | `parking×2 + balconies×1` |
| `total_rooms` | `bedrooms + bathrooms` |

---

## 🤖 Models Used

| Model | Strength | Weakness |
|---|---|---|
| Linear Regression | Fast, interpretable | Assumes linearity |
| Decision Tree | Non-linear, explainable | Prone to overfitting |
| Random Forest | Robust, handles noise | Slower, less interpretable |
| XGBoost | State-of-art accuracy | Needs hypertuning |

---

## 📈 Results

> Results from a sample run (may vary slightly):

| Model | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | R² Score |
|---|---|---|---|
| Linear Regression | ~18 | ~28 | ~0.72 |
| Decision Tree | ~14 | ~22 | ~0.82 |
| Random Forest | ~10 | ~16 | ~0.91 |
| **XGBoost** | **~9** | **~14** | **~0.93** |

**Best Model:** XGBoost | **R² ≈ 93%** accuracy

---

## 📁 Folder Structure

```
House-Price-Prediction/
│
├── data/
│   └── housing_data.csv          ← Generated dataset
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py         ← Synthetic data creation
│   ├── preprocessing.py          ← Cleaning + encoding + scaling
│   ├── train.py                  ← Model training + saving
│   ├── evaluate.py               ← Metrics + cross-validation
│   ├── visualize.py              ← 8 EDA + evaluation charts
│   └── predict.py                ← Interactive CLI prediction
│
├── models/
│   ├── best_model.pkl            ← Saved best model
│   ├── encoders.pkl              ← Label encoders
│   ├── scaler.pkl                ← StandardScaler
│   └── all_models.pkl            ← All trained models
│
├── outputs/
│   ├── model_comparison.csv      ← Side-by-side metrics table
│   ├── test_predictions.csv      ← Actual vs Predicted values
│   └── evaluation_report.csv    ← Full evaluation metrics
│
├── images/
│   ├── 01_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_by_location.png
│   ├── 04_area_vs_price.png
│   ├── 05_actual_vs_predicted.png
│   ├── 06_residuals.png
│   ├── 07_feature_importance.png
│   └── 08_model_comparison.png
│
├── notebooks/
│   └── EDA_and_Modelling.ipynb   ← Jupyter exploration notebook
│
├── main.py                       ← Full pipeline orchestrator
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## ⚙️ Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the full pipeline (train + evaluate + visualize + predict)
```bash
python main.py
```

### Run prediction only (use saved model)
```bash
python main.py --predict-only
```

### Run individual modules
```bash
python src/data_generator.py     # Generate dataset only
python src/train.py              # Train models only
python src/evaluate.py           # Evaluate best model
python src/visualize.py          # Generate all charts
python src/predict.py            # Interactive prediction CLI
```

### Expected Terminal Output
```
══════════════════════════════════════════════════════════════
  STEP 1 & 2 — Data Generation + Model Training
══════════════════════════════════════════════════════════════
📦 Generating / loading dataset ...
⚙️  Preprocessing ...
🏋️  Training models ...
  ✅ Linear Regression trained in 0.1s
  ✅ Decision Tree trained in 0.3s
  ✅ Random Forest trained in 4.2s
  ✅ XGBoost trained in 6.8s

🏆 Best model: XGBoost (R² = 0.9312)

══════════════════════════════════════════════════════════════
  STEP 5 — Demo Prediction (3BHK Andheri, 1200 sqft)
══════════════════════════════════════════════════════════════
  🏡  Predicted Price  : ₹ 87.43 Lakhs
```

---

## 🖼️ Screenshots

| Chart | Description |
|---|---|
| `images/01_price_distribution.png` | Raw + log price histogram |
| `images/02_correlation_heatmap.png` | Feature correlation matrix |
| `images/03_price_by_location.png` | Price spread per location |
| `images/04_area_vs_price.png` | Area vs Price coloured by bedrooms |
| `images/05_actual_vs_predicted.png` | Model accuracy scatter |
| `images/06_residuals.png` | Error distribution analysis |
| `images/07_feature_importance.png` | Top driving features |
| `images/08_model_comparison.png` | MAE / RMSE / R² comparison |

---

## 🎓 Key Learnings

- How to build a complete ML pipeline from scratch
- Synthetic data generation that mirrors real-world distributions
- Feature engineering creates more predictive power than raw features
- Ensemble models (Random Forest, XGBoost) significantly outperform linear models
- Residual analysis reveals model weaknesses
- Joblib serialization for model deployment
- `StandardScaler` must be fitted only on training data (data leakage prevention)

---

## 💼 Interview Q&A

**Q: Why use Random Forest over Linear Regression for house prices?**
> House prices have non-linear relationships with features. Random Forest captures these interactions via multiple decision trees, giving far better accuracy.

**Q: What is R² score and what does 0.93 mean?**
> R² measures how much variance in the target the model explains. 0.93 means the model explains 93% of price variability — excellent for a regression problem.

**Q: How do you prevent data leakage in the pipeline?**
> By fitting the StandardScaler and LabelEncoders only on training data, and using `transform` (not `fit_transform`) on the test set.

**Q: What is overfitting and how do you detect it?**
> When a model performs well on training data but poorly on unseen data. We detect it via cross-validation — if CV score ≈ training score, no overfitting.

**Q: How would you deploy this model in production?**
> Save the model with Joblib, wrap it in a FastAPI or Flask endpoint, containerize with Docker, and deploy on AWS/GCP with a CI/CD pipeline.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built with ❤️ as a portfolio project demonstrating end-to-end Machine Learning engineering skills.

⭐ **Star this repo if it helped you!**
#   H o u s e - P r i c e - P r e d i c t i o n -  
 