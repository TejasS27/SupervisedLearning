import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_excel("SimpleLinearRegression\\McDonalds\\Mc_Donalds.xlsx")
print(df.head(5))

# Checking for duplicates if any
print(df.duplicated().sum())

# Checking for null values if any
print(df.isnull().sum())

# Checking for outliers - we plot box plot
sns.set_style("whitegrid")

plt.figure(figsize=(12,5))

# For New Hourly Wage
plt.subplot(1,2,1)
sns.boxplot(data=df, y = "New Hourly Wage", color="skyblue")
plt.title("Box Plot - New Hourly Wage")

# Box plot for 'Big Mac Price'
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Big Mac Price', color='lightgreen')
plt.title('Box Plot - Big Mac Price')

plt.show()

# Declaring x and y variable
X = df['Big Mac Price']
y  = df['New Hourly Wage']

# Check for correlation between Big Mac Price and New Hourly Price
correlation = df['Big Mac Price'].corr(df['New Hourly Wage'])
print(f"The correlation between Big Mac Price and New Hourly Wage is: {correlation}")


# Add a constant to the independent variable for the intercept term
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Predict the net hourly wage based on Big Mac prices
predictions = model.predict(X)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(df['Big Mac Price'], df['New Hourly Wage'], label='Actual')
plt.plot(df['Big Mac Price'], predictions, color='red', label='Predicted')
plt.xlabel('Big Mac Price')
plt.ylabel('New Hourly Wage')
plt.title('Actual vs. Predicted Net Hourly Wage')
plt.legend()
plt.show()

# Scatter Plot of Residuals to check Homoscedasticity

residuals = model.resid

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Scatter Plot of Residuals')
plt.show()

# Predict new hourly wage for a Big Mac price of 3

new_big_mac_price = 3
new_hourly_wage = -4.4704 + 4.7029 * new_big_mac_price
print(f"Predicted New Hourly Wage for a Big Mac Price of ${new_big_mac_price}: ${new_hourly_wage:.2f}")

