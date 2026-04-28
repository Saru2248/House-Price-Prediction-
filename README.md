
# 🏠 House Price Prediction using Machine Learning

> A complete end-to-end Machine Learning project that predicts residential property prices using regression models. Built with synthetic real-estate data to simulate real-world pricing systems.

---

## 📌 Project Overview

House price prediction is a **supervised regression problem** where the goal is to estimate the market value of a property based on its features such as area, number of rooms, location, and amenities.

This project replicates a real-world pricing pipeline used in:
- Property listing platforms
- Banking loan valuation systems
- Real estate analytics tools

---

## ❓ Problem Statement

Property pricing is often inconsistent and subjective.

**Goal:**  
Build a data-driven system that predicts house prices accurately using historical property features.

---

## 🏭 Industry Relevance

This type of system is widely used in:

- Real estate platforms like MagicBricks and 99acres  
- Banks for loan approval and collateral valuation  
- Investors for identifying undervalued properties  

---

## ⚙️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Models | Linear Regression, Random Forest |

---

## 📊 Dataset

- Synthetic dataset (simulating real housing data)
- 500+ records
- Features include:

| Feature | Description |
|--------|------------|
| area | Size of house (sq ft) |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| age | Age of property |
| parking | Parking availability |
| location | Categorical location |

---

## 🧠 Models Used

### 1. Linear Regression
- Simple baseline model
- Assumes linear relationships

### 2. Random Forest Regressor
- Handles non-linearity
- More accurate and robust

---

## 📈 Model Evaluation

Metrics used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Observation:**
- Random Forest outperformed Linear Regression
- Better handling of complex relationships

---

## 🏗️ Project Architecture

```

Input Features
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Prediction Output
↓
Visualization

```

---

## 📁 Folder Structure

```

House-Price-Prediction/
│
├── data/          # Dataset
├── notebooks/     # Jupyter notebooks
├── src/           # Core scripts
├── models/        # Saved models
├── outputs/       # Predictions & results
├── images/        # Graphs & charts
├── main.py        # Main execution file
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

```bash
git clone https://github.com/Saru2248/house-price-prediction.git
cd house-price-prediction
````

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Output

* Model performance metrics printed in terminal
* Predicted house price for sample input
* Visualization saved in `/outputs` folder

---

## 🖼️ Key Visualizations

* Correlation Heatmap
* Actual vs Predicted Prices
* Feature Importance (Random Forest)
* Price Distribution

---

## 🧪 Sample Prediction

```
Input:
Area = 2000 sq ft
Bedrooms = 3
Bathrooms = 2
Age = 5 years

Output:
Predicted Price ≈ ₹ X Lakhs
```

---

## 🎯 Key Learnings

* Data preprocessing and feature engineering
* Regression model comparison
* Model evaluation techniques
* Handling categorical data
* Real-world ML workflow

---

## 💼 Interview Talking Points

* Why Random Forest performs better than Linear Regression
* Bias vs Variance trade-off
* Importance of feature engineering
* Evaluation metrics for regression

---

## 🚀 Future Improvements

* Use real dataset (Kaggle / Housing data)
* Add XGBoost model
* Deploy using Flask or Streamlit
* Build UI for user input

---

## 👨‍💻 Author

Sarthak Dhumal
Aspiring Data Scientist / ML Engineer

---

## ⭐ If you found this useful, consider giving it a star!

```

