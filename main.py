import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load and preprocess data
data_set = pd.read_csv('AI Research & Development Fellowship - Assessment Datasets - Question 3 - Statistical Reasoning.csv')
data_set['Date'] = pd.to_datetime(data_set['Date'])
data_set['Year'] = data_set['Date'].dt.year
data_set['Month'] = data_set['Date'].dt.month

x = data_set[['Year', 'Month']]
y = data_set['Revenue']
x = sm.add_constant(x)

# Fit model
model = sm.OLS(y, x).fit()
print(model.summary())

# Scatter Plots with Regression Line
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=data_set['Year'], y=data_set['Revenue'], label='Actual')
sns.regplot(x=data_set['Year'], y=data_set['Revenue'], scatter=False, color='red', label='Fitted Line')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Year vs Revenue')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=data_set['Month'], y=data_set['Revenue'], label='Actual')
sns.regplot(x=data_set['Month'], y=data_set['Revenue'], scatter=False, color='red', label='Fitted Line')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Month vs Revenue')
plt.legend()

plt.tight_layout()
plt.show()

# Residuals Plot
data_set['Residuals'] = model.resid
plt.figure(figsize=(8, 6))
sns.scatterplot(x=model.fittedvalues, y=data_set['Residuals'])
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(data_set['Residuals'], kde=True)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.show()

# QQ Plot
plt.figure(figsize=(8, 6))
stats.probplot(data_set['Residuals'], dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

# Predicted vs Actual Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=model.fittedvalues, y=data_set['Revenue'])
plt.plot(data_set['Revenue'], data_set['Revenue'], color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual Values')
plt.show()
