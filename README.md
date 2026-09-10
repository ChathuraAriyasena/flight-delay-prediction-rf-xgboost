# ✈️ Data-Driven Insights & Flight Delay Prediction — U.S. Airline Industry

**Machine Learning | Predictive Analytics | Business Intelligence**

A full-cycle data science project that analyzes 500K+ U.S. domestic flight records to uncover delay drivers and predict both **flight delay likelihood** and **delay duration**, using `RandomForestClassifier` and `XGBoostRegressor`. Built to demonstrate how predictive modeling can translate into measurable operational and financial outcomes for the airline industry.

📄 **Full Report:** [Data-Driven Insights and Flight Predictions Using Airline Data (PDF)](./Data-Driven-Insights-and-Flight-Predictions-Using-Airline-Data.pdf)
📓 **Notebook:** [CB015565_COIS71203-0210-Business_Analytics-2024.ipynb](./CB015565_COIS71203-0210-Business_Analytics-2024.ipynb)

---

## 📌 Business Problem

Flight delays cost the global aviation industry an estimated **$26 billion annually**, driven by fuel burn, crew overtime, and passenger compensation. U.S. airlines' direct operating cost sits at **$100.80 per block minute** (Airlines for America, 2024). Beyond cost, delays erode customer satisfaction, loyalty, and brand reputation.

**Objective:** Use machine learning to predict delays before they escalate — enabling airlines to proactively reallocate resources, adjust scheduling, and improve on-time performance and customer experience.

---

## 🧰 Tech Stack & Keywords

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `imbalanced-learn (SMOTE)` · `Google Colab` · `Jupyter Notebook`

**Techniques:** Machine Learning · Predictive Modeling · Data Preprocessing · Data Cleaning · Exploratory Data Analysis (EDA) · Feature Engineering · Outlier Detection (IQR Method) · Label Encoding · One-Hot Encoding · Classification · Regression · Random Forest · Gradient Boosting (XGBoost) · Gini Impurity · Hyperparameter Tuning · Grid Search · Cross-Validation · Early Stopping · Confusion Matrix · ROC-AUC Curve · Precision/Recall/F1-Score · R² & RMSE Evaluation · Data Visualization · Data Storytelling · Business Analytics · Operational Efficiency · Digital Transformation · Automation

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Source | U.S. Department of Transportation (via Kaggle) |
| Rows (raw) | 504,446 |
| Rows (post outlier removal) | 404,694 |
| Columns | 27 |
| Missing Values | None |
| Train/Test Split | 80% / 20% (~323,755 / 80,939 rows) |

Key fields: flight number, airline code, origin/destination airports, scheduled vs. actual departure/arrival times, and categorized delay causes (air system, security, airline, late aircraft, weather, check/immigration).

---

## 🔧 Methodology

### 1. Data Preprocessing
- Merged airline reference data (IATA code → airline name) via lookup join
- Removed duplicates and irrelevant fields (tail number, wheels-off/on, air time, etc.)
- Handled outliers in `DEPARTURE_DELAY` using the **Interquartile Range (IQR)** method across iterative passes
- Encoded categorical variables (`LabelEncoder`, `pd.get_dummies`) for model readiness

### 2. Exploratory Data Analysis & Visualization
- Delay proportion analysis (75.8% delayed vs. 24.2% on-time)
- Delay trend analysis by scheduled departure time, day of week, and airline
- Categorical breakdown of delay causes (air system, weather, security, airline, late aircraft, check/immigration)
- Heatmaps, boxplots, histograms, and time-series charts built with `Matplotlib` and `Seaborn`

### 3. Predictive Modeling

**Model 1 — RandomForestClassifier** (Flight Status Prediction: Delayed vs. On-Time)
- Binary classification using Gini impurity criterion, 150 estimators
- Predictors: day, day of week, flight number, airline, scheduled times, origin/destination airport

**Model 2 — XGBoostRegressor** (Delay Duration Prediction in Minutes)
- Regression on continuous delay magnitude with L1/L2 regularization (alpha, lambda)
- Tuned learning rate, max depth, subsample, and colsample_bytree
- Early stopping (50 rounds) across 2,000 boosting rounds to prevent overfitting

### 4. Model Tuning
- Grid search and cross-validation for hyperparameter optimization
- Regularization and early stopping to control overfitting and improve generalizability

---

## 📈 Results

### RandomForestClassifier — Flight Status Prediction

| Metric | Score |
|---|---|
| Accuracy | **0.77** |
| ROC-AUC | **0.74** |
| Precision (Delayed) | 0.81 |
| Recall (Delayed) | 0.91 |
| F1-Score (Delayed) | 0.86 |

### XGBoostRegressor — Delay Duration Prediction

| Metric | Train | Test |
|---|---|---|
| MSE | 48.39 | 59.57 |
| R² | 0.60 | 0.50 |

---

## 🏭 Production Use Case

Designed as a deployable prediction service: an analyst or operations team submits flight details (date, airline, scheduled times, origin/destination), and the system returns:
1. **Predicted flight status** (Delayed / On-Time) via RandomForestClassifier
2. **Predicted delay duration in minutes** via XGBoostRegressor

Proposed architecture: expose trained models as a **web service via middleware**, accepting **JSON/XML requests** and returning real-time predictions for integration into airline operations dashboards or front-end applications — supporting proactive resource allocation, schedule adjustment, and passenger communication.

---

## 💡 Business Impact & Key Takeaways

- Identified **check/immigration delay (77%)** and **airline-caused delay (53.3%)** as the most frequent disruption categories, providing actionable insight for operational prioritization
- Demonstrated how classification and regression models can be paired to answer two distinct business questions: *"Will this flight be delayed?"* and *"By how much?"*
- Translated raw operational data into a decision-support tool aligned with reducing costs, improving on-time performance, and enhancing customer satisfaction — bridging **data science and business analysis**

---

## ⚠️ Limitations & Future Work

- Feature engineering and imputation are foundational; predictive imputation and interaction terms could improve accuracy
- Limited hyperparameter search space and no k-fold cross-validation reported in final results
- Future iterations could incorporate weather API data, real-time flight tracking feeds, and ensemble stacking (Random Forest + XGBoost + Logistic Regression)

---

## 📚 References

1. Cook, A., Tanner, G., Williams, V., & Meise, G. (2015). *Dynamic cost indexing – Managing airline delay costs.* Journal of Air Transport Management, 47, 32-38.
2. Jiang, H., & Zhang, Y. (2016). *An analysis of the impact of flight delays on customer satisfaction.* Transportation Research Part A, 91, 197-208.
3. Barbot, C. (2006). *Low-cost airlines, secondary airports, and state aid.* Journal of Air Transport Management, 12(4), 197-203.
4. Belobaba, P., Odoni, A., & Barnhart, C. (2015). *The Global Airline Industry.* John Wiley & Sons.
5. Tsoukalas, G., Belobaba, P. P., & Swelbar, W. (2008). *Modeling the impact of airline schedule complexity on operational performance.* Transportation Research Part A, 42(5), 731-743.
6. Airlines for America. (2024). *U.S. Passenger Carrier Delay Costs.* [airlines.org](https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs)

---

## 👤 Author

**Chathura Ariyasena**
Lead Business Analyst | Digital Transformation & AI-Driven Automation
Faculty of Computing, University of Staffordshire (APIIT Sri Lanka)
📧 chathura.ariyasena@outlook.com

*This project was completed as part of a Business Analytics program and is showcased here to demonstrate applied machine learning, data storytelling, and analytical skills relevant to business analysis, digital transformation, and AI initiatives.*
