# Multiple Linear Regression"

# Prediction of Energy Consumption of Appliances 

This project explores the relationship between **Appliances Energy Consumption** and various **Temperature**  and **Environmental** variable factors.

---

# OLS Regression Analysis: Energy Consumption of Appliances

### Project Overview

This project performs **Ordinary Least Squares (OLS) Regression** to model and analyze the energy consumption (`Appliances`) of a household using environmental and weather-related variables. The analysis is based on a dataset containing approximately 19,735 observations and 26 features.

---

### Files

- `Scaled_Final_Regression.ipynb`: Main Jupyter Notebook containing data preprocessing, model fitting, evaluation, and residual diagnostics.
- `README.md`: This file.

---

### Dataset Summary

- **Target Variable**: `Appliances` (Energy consumption in Wh)
- **Independent Variables**: Indoor temperatures (T1-T9), humidity levels (RH_1 to RH_9), lights, outdoor weather data (`T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`), and others like `rv1`, `rv2`.
- **Observations**: 19,735
- **Features**: 26

---

### Methodology

### Data Preprocessing:
- Standardized independent variables.
- Scaled features to improve model interpretability.

### Model Used:
- **Ordinary Least Squares (OLS)** from `statsmodels`.

---

### Regression Results Summary

- **R-squared**: 0.290  
  Indicates the model explains ~29% of the variance in appliance energy usage.

- **Adjusted R-squared**: 0.289  
  Slightly penalizes for additional variables, suggesting some may not significantly contribute.

- **F-statistic**: 309.2 (p < 0.001)  
  Overall model is statistically significant.

---

### Significant Predictors (p < 0.05)

| Feature | Coefficient | Interpretation |
|--------|-------------|----------------|
| `lights` | +1.076 | Higher light usage increases appliance energy use. |
| `RH_1`, `RH_3`, `RH_5`, `RH_6` | Positive impact | Indicates humidity at certain indoor zones influences consumption. |
| `T3`, `T6`, `T8` | Positive impact | Higher room temperatures in some areas correspond with higher energy use. |
| `T2`, `T4`, `T7`, `T9`, `T_out` | Negative impact | Suggests temperature differences reduce appliance load (possibly due to better insulation or lower heating/cooling needs). |
| `Windspeed` | Positive impact | May relate to HVAC compensation. |

---

### Model Diagnostics

### 1. **Heteroscedasticity**
- **Breusch-Pagan or visual test (notebook not shown here)** confirms heteroscedasticity.
- **Conclusion**: Error variance is not constant; violates one of the key OLS assumptions.
- **Action**: Consider robust standard errors, Generalized Least Squares (GLS), or transformation.

### 2. **Multicollinearity**
- Warning: *"The smallest eigenvalue is close to zero"*, suggesting **strong multicollinearity**.
- `rv1` and `rv2` are likely redundant (identical coefficients).
- **Action**: Remove or combine collinear features (e.g., via PCA or VIF analysis).

### 3. **Autocorrelation**
- **Durbin-Watson = 0.482** → Indicates strong **positive autocorrelation**.
- **Action**: Use time-series models like ARIMA or incorporate lag variables if temporal structure is important.

---

### Conclusion & Recommendations

- The model captures some meaningful relationships, especially around temperature, humidity, and lighting.
- However, the **low R²**, presence of **heteroscedasticity**, **multicollinearity**, and **autocorrelation** limit reliability.
- Future models should consider:
  - Feature selection or dimensionality reduction.
  - Robust regression techniques.
  - Exploring non-linear models (e.g., Random Forest, Gradient Boosting).

---

