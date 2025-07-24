# Turnover-Trend-Forecaster


Employee Turnover Prediction Project Documentation

Project Summary

This project developed a predictive model to identify employees at risk of leaving an organization, leveraging a dataset of approximately 1,470 employees. The target variable, Attrition (0 for stay, 1 for leave), revealed a class imbalance with ~83.88% stayers and ~16.12% leavers (1,233 stayed, 237 left). The analysis began with data visualization, uncovering patterns such as longer DistanceFromHome and lower MonthlyIncome correlating with higher turnover. Data preprocessing involved encoding categorical features (e.g., JobRole, Gender) using OneHotEncoder, scaling numerical features with MinMaxScaler, and splitting the data into 75% training (1,102 samples) and 25% testing (368 samples).
Three classifiers were employed: Logistic Regression, Random Forest, and Artificial Neural Network (ANN). The Logistic Regression model achieved 89.9% accuracy, with a confusion matrix of [[300, 7], [30, 26]] and an F1-score of 0.58 for leavers (recall 0.46). The Random Forest model reached ~90% accuracy, with a confusion matrix of [[310, 0], [44, 17]], precision ~100% for leavers, but a recall of only 0.21 (F1-score ~0.35). The ANN, built with TensorFlow/Keras, achieved ~84% accuracy, with a confusion matrix of [[300, 24], [31, 25]] and an F1-score of 0.48 for leavers (recall 0.45). All models performed strongly for stayers (F1-score ~0.91–0.94) but struggled with leavers due to the imbalance, missing 30–44 leavers. This limitation could cost ~$228,000–$235,600 (30–31 × $7,600 per hire). Key drivers like DistanceFromHome, JobRole, and JobSatisfaction suggest HR interventions such as remote work or role adjustments.
Objective
The objective is to predict which employees are likely to leave based on factors including job involvement, education, job satisfaction, performance, relationship satisfaction, and work-life balance. By identifying at-risk employees and understanding turnover drivers, the company can implement targeted retention strategies to reduce costs and improve workforce stability.
Business Case
Employee turnover incurs significant financial and operational costs:

Hiring Burden: Small businesses dedicate 40% of working hours to non-revenue tasks like hiring, with an average of 52 days to fill a position.
Recruitment Expenses: Replacement costs average $7,600 per employee, or 15-20% of an employee’s salary (e.g., $15,000–$20,000 for a $100,000 salary).
Revenue Impact: Onboarding new hires reduces revenue by 1-2.5% as they adapt to systems and teams.
Knowledge Loss: Departing employees take expertise, disrupting operations and straining remaining staff.

Benefits of Prediction:

Cost Reduction: Proactive retention avoids recruitment and onboarding expenses.
Retention Enhancement: Insights improve job satisfaction, work-life balance, and compensation.
Talent Preservation: Targeted efforts (e.g., bonuses, training) retain high-risk, high-value employees.

This project equips HR with data-driven tools to save money, boost morale, and maintain competitiveness.
Methodology
Import Libraries and Dataset
Libraries

Pandas: Dataframe manipulation (e.g., loading Human_Resources.csv).
NumPy: Numerical computations.
Seaborn/Matplotlib: Data visualization (e.g., histograms, KDE plots).
Scikit-learn: Machine learning tools (e.g., OneHotEncoder, MinMaxScaler).
TensorFlow/Keras: ANN development.

Dataset Loading

Loaded Human_Resources.csv from Google Drive into employee_df (1,470 rows, 35 columns).
Features include Age, JobSatisfaction, MonthlyIncome, DistanceFromHome, and Attrition.

Initial Exploration

info Method: Confirmed 1,470 rows, 35 columns (numerical: int64; categorical: object), no missing values.
describe Method: Mean Age ~37, MonthlyIncome ~$6,500, DistanceFromHome ~9 km, suggesting a mature workforce with potential turnover predictors.

Data Visualization
Preprocessing for Visualization

Transformed categorical columns (Attrition, OverTime, Over18) to numerical (0/1) using lambda functions.
Confirmed no missing data via Seaborn heatmap.
Dropped irrelevant columns (EmployeeCount, StandardHours, Over18, EmployeeNumber), reducing features to 31.

Visualizations

Histograms: Explored distributions (e.g., Age 30–40, MonthlyIncome $0–$5,000 tail-heavy).
Attrition Split: Created left_df (237 leavers) and stayed_df (1,233 stayers).
Correlations:
TotalWorkingYears vs. JobLevel (0.78): Experience drives seniority.
TotalWorkingYears vs. MonthlyIncome (0.77): Experience increases pay.
YearsAtCompany vs. MonthlyIncome (0.51): Loyalty boosts income.


Count Plots:
Age: High turnover at 28–31; low at 53–60.
JobRole: Sales Representatives (50% turnover), Research Directors (6% turnover).
MaritalStatus: Single (22% turnover) vs. married/divorced (lower).
JobInvolvement: Low involvement linked to leaving.


KDE Plots:
DistanceFromHome: Higher turnover at 10–30 km.
YearsWithCurrManager: Turnover higher at 0–5 years.
TotalWorkingYears: Turnover higher at 0–7 years.


Box Plots:
MonthlyIncome vs. Gender: Similar medians, slight female max advantage.
MonthlyIncome vs. JobRole: Managers ($17,000 median) vs. Sales Representatives (lowest).



Insights

Younger, single, entry-level employees in low-paying, low-involvement roles (e.g., Sales Representatives) with long commutes are more likely to leave.
Older, senior, engaged employees with higher incomes and shorter commutes tend to stay.

Data Cleaning and Preprocessing

Categorical Features: Separated six features (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus) into X_cat, encoded with OneHotEncoder (26 columns).
Numerical Features: Isolated numerical columns (e.g., Age, DailyRate) into X_numerical.
Combined Features: Concatenated X_cat and X_numerical into X_all (50 columns).
Scaling: Applied MinMaxScaler to normalize features to [0, 1].
Target Variable: Defined y as Attrition (1,470 binary values).

Creating Training and Testing Datasets

Train-Test Split: 75% training (1,102 samples, 50 features), 25% testing (368 samples, 50 features).

Model Development and Intuition
Classifiers

Logistic Regression:
Linear model with sigmoid function for binary classification.
Interpretable coefficients highlight feature influence (e.g., MonthlyIncome).


Random Forest:
Ensemble of decision trees, voting on outcomes.
Robust to noise, ranks feature importance (e.g., DistanceFromHome).


Artificial Neural Network (ANN):
Feedforward network with input (50 features), hidden (500 units, ReLU), and output (1 unit, sigmoid) layers.
Captures complex, non-linear patterns.



Evaluation Metrics

Confusion Matrix: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
KPIs:
Accuracy: (TP + TN) / Total.
Precision: TP / (TP + FP).
Recall: TP / (TP + FN).
F1-Score: 2 × (Precision × Recall) / (Precision + Recall).



Model Training and Evaluation
Logistic Regression

Training: Fit on X_train, y_train.
Results:
Accuracy: 89.9%.
Confusion Matrix: [[300, 7], [30, 26]].
Precision: 0.91 (stay), 0.79 (leave).
Recall: 0.98 (stay), 0.46 (leave).
F1-Score: 0.94 (stay), 0.58 (leave).


Interpretation: Strong for stayers, misses 30 of 61 leavers.

Random Forest

Training: Fit on X_train, y_train.
Results:
Accuracy: ~90%.
Confusion Matrix: [[310, 0], [44, 17]].
Precision: ~1.00 (stay), ~1.00 (leave).
Recall: ~1.00 (stay), 0.21 (leave).
F1-Score: ~1.00 (stay), ~0.35 (leave).


Interpretation: Excellent precision, poor recall (misses 44 of 61 leavers).

Artificial Neural Network (ANN)

Setup: Two hidden layers (500 units, ReLU), output layer (sigmoid), Adam optimizer, binary cross-entropy loss.
Training: 100 epochs, batch size 50, accuracy rose to ~100% on training data.
Results:
Accuracy: ~84%.
Confusion Matrix: [[300, 24], [31, 25]].
Precision: 0.90 (stay), 0.51 (leave).
Recall: 0.93 (stay), 0.45 (leave).
F1-Score: 0.91 (stay), 0.48 (leave).


Interpretation: Moderate performance, misses 31 of 56 leavers.

Discussion

Class Imbalance: All models favor stayers (~83.88%), with low recall for leavers (0.21–0.46), missing 30–44 leavers. This could cost ~$228,000–$235,600 ($7,600/hire).
Key Drivers: DistanceFromHome (10–30 km), low MonthlyIncome (e.g., Sales Representatives), and low JobSatisfaction drive turnover.
Interventions: Remote work, pay adjustments, and engagement programs could reduce attrition.

Recommendations

Model Improvement: Apply class weighting or SMOTE to address imbalance and boost recall.
Feature Engineering: Explore interactions (e.g., Age × JobSatisfaction).
Deployment: Integrate into HR systems for real-time monitoring.
HR Actions: Target high-risk groups with remote work, raises, or role enhancements.

This project demonstrates predictive analytics’ value in HR, offering actionable insights to enhance retention and minimize turnover costs.
