import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
df = pd.read_csv("MultipleLinearRegression\\EnergyConsumption\\Appliance_Energy.csv")

print(df.head(5))

# Drop non-numeric or unnecessary columns
df = df.drop(columns=['date'])

# Checking null values if any
df.isnull().sum()

# Checking duplicates value if any
df.duplicated().sum()

plt.figure(figsize=(10,10))
df.boxplot()
plt.show()

# Get percentage of outliers


# Assuming 'data' DataFrame is already loaded and scaled as in the provided code

# Select numerical columns for outlier detection (excluding non-numerical columns if any)
numerical_cols = df.select_dtypes(include=['number']).columns

# Calculate the percentage of outliers for each numerical column
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_sum = len(outliers) / len(df)

    print(f"Outliers of variable {col}: {outlier_sum:.2f}")

# Clear Outliers using Inter quartile Range

# Assuming 'data' DataFrame is already loaded and scaled as in the provided code

# Select numerical columns for outlier treatment (excluding 'lights' and non-numerical columns)
numerical_cols = df.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols.drop('lights')

# Treat outliers using IQR for selected numerical columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the nearest bound
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Re run the outliers to check whether they are fully nullified.
numerical_cols = df.select_dtypes(include=['number']).columns

# Calculate the percentage of outliers for each numerical column
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_sum = len(outliers) / len(df)

    print(f"Outliers of variable {col}: {outlier_sum:.2f}")

# Calculate the correlation matrix
corr_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Define features and target

# Define features (X) and target (y)
X = df.drop('Appliances', axis=1)
y = df['Appliances']
print(X.head(5))
print(y.head(5))

# Normalize features using standard scalar

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Perform mlr using stats model


# Add a constant to the independent variables
X_scaled = sm.add_constant(X_scaled)

# Fit the multiple linear regression model
model = sm.OLS(y, X_scaled).fit()

# Print the model summary
print(model.summary())

# Fit initial model
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# Identify insignificant variables
insignificant_vars = [var for var, pval in model.pvalues.items() if pval > 0.05]

# Filter only those that exist in X
if 'const' in insignificant_vars:
    insignificant_vars.remove('const') #Constant term should not be removed

# Refit the model
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# Print updated summary
print(model.summary())

# Check for homoscedasticity using white test

# Perform the White test
white_test = sms.het_white(model.resid, model.model.exog)

# Print the test results
white_test

# Interpret the results (p-value)
p_value = white_test[1]  # Access the p-value from the test results
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: Homoscedasticity is present.")

# Generate plots for checking homoscedasticity

import matplotlib.pyplot as plt
# Assuming 'data' DataFrame and 'model' are already defined as in the previous code

# Plot residuals vs. fitted values
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values Plot")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.show()