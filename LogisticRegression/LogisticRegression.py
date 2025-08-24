import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn  as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("logistic_regression.csv")
print(df.head(5))


# Data Cleaning
print(df.isnull().sum())


# Identified and separated dependent (y) and independent (X) variables
X = df[['Studied', 'Slept']]
y = df['Passed']

# Split data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Relationship (Collinearity)
sns.pairplot(df, hue='Passed')
plt.show()

# Correlation Matrix
print("Correlation matrix:\n", df.corr())

# Variance Inflation Factor (VIF) check for multi-collinearity
X_vif = df[['Studied', 'Slept']]
X_vif_constant = sm.add_constant(X_vif)

vif_df = pd.DataFrame()
vif_df['Feature'] = X_vif_constant.columns
vif_df['VIF'] = [variance_inflation_factor(X_vif_constant.values, i) for i in range(X_vif_constant.shape[1])]
print("VIF values:\n", vif_df)

# Initialize and fit data in Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict output
y_predicted = model.predict(X_test)
y_predicted_pr = model.predict_proba(X_test)[:,1]

# Metrics
print('Coefficients : ', model.coef_)
print('Intercept : ', model.intercept_)
print('Accuracy Score : ', accuracy_score(y_test, y_predicted))

# Model Diagnostics
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_predicted))

roc_auc = roc_auc_score(y_test, y_predicted_pr)
print("ROC-AUC Score:", roc_auc)