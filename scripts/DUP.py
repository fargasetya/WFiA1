#DUP Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

df=pd.read_csv(r'C:/Users/reube/Downloads/WFiA1/weatherAUS.csv')

#change date to date time
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df.sort_values(by='Date', inplace=True)

# Replace NA values with mean for numeric columns
numeric_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 
                   'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
                   'Pressure3pm', 'Temp9am', 'Temp3pm']

for col in numeric_columns:
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)
    
# Remove unnecessary columns
columns_to_remove = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Cloud9am', 'Cloud3pm']
df.drop(columns=columns_to_remove, inplace=True)    

print(df.columns)

# Create a new column 'Rain_in_past_5_days'
df['Rain_in_past_5_days'] = df['Rainfall'].rolling(window=5, min_periods=1).sum()

# If the sum of rainfall in the past 5 days is greater than 0, set 'Rain_in_past_5_days' to 1, otherwise, set it to 0
df['Rain_in_past_5_days'] = (df['Rain_in_past_5_days'] > 0).astype(int)

rain_distribution = df['RainTomorrow'].value_counts()
plt.bar(rain_distribution.index, rain_distribution.values, color=['skyblue', 'lightcoral'])
plt.title('Distribution of Rain Tomorrow')
plt.xlabel('Rain Tomorrow')
plt.ylabel('Count')
plt.show()

numeric_columns = df.select_dtypes(include=['float64', 'int32']).columns

# Exclude object (string) columns from correlation calculation
numeric_df = df[numeric_columns]

# Calculate correlation matrix for numeric columns
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
df['WindDir3pm'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Wind Direction at 3pm Distribution')
plt.xlabel('Wind Direction')
plt.ylabel('Count')
plt.show()

plt.scatter(df['Temp9am'], df['Temp3pm'], c=df['RainTomorrow'].map({'No': 'blue', 'Yes': 'red'}).fillna('gray'), alpha=0.5)
plt.title('Scatter Plot of Temperature (9am vs. 3pm) with RainTomorrow')
plt.xlabel('Temperature at 9am')
plt.ylabel('Temperature at 3pm')
plt.show()

# Select numeric columns for correlation
numeric_columns = df.select_dtypes(include=['float64', 'int32']).columns

# Exclude non-numeric columns from correlation calculation
numeric_df = df[numeric_columns]

# Drop rows with any NaN values in the numeric columns
numeric_df = numeric_df.dropna()

correlation_matrix = numeric_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Rain_in_past_5_days'], label='Rain in Past 5 Days', color='blue')
plt.title('Rain in Past 5 Days - Line Plot')
plt.xlabel('Date')
plt.ylabel('Rain in Past 5 Days')
plt.legend()
plt.show()

df = df.dropna(subset=['RainTomorrow'])

# Define features (X) and target variable (y)
X = df[['Rain_in_past_5_days']]
y = df['RainTomorrow']
df_majority = df[df['RainTomorrow'] == 'No']
df_minority = df[df['RainTomorrow'] == 'Yes']

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df.set_index('Date', inplace=True)

# Create and fit the SARIMA model
order = (1, 1, 1) 
seasonal_order = (1, 1, 1, 12)
sarima_model = SARIMAX(train_data['Rain_in_past_5_days'], order=order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

# Forecasting
forecast = sarima_fit.get_forecast(steps=len(test_data))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plotting the forecast
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data['Rain_in_past_5_days'], label='Training data', color='blue')
plt.plot(test_data.index, test_data['Rain_in_past_5_days'], label='Actual test data', color='green')
plt.plot(test_data.index, forecast_mean, label='SARIMA forecast', color='red')
plt.fill_between(test_data.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Rain in Past 5 Days')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test_data['Rain_in_past_5_days'], forecast_mean)
print(f'Mean Squared Error (MSE): {mse}')

# Create and fit the SARIMAX model
sarimax_model = SARIMAX(train_data['Rain_in_past_5_days'], exog=exog_train, order=order, seasonal_order=seasonal_order)
sarimax_fit = sarimax_model.fit(disp=False)

# Forecasting
forecast_sarimax = sarimax_fit.get_forecast(steps=len(test_data), exog=exog_test)
forecast_mean_sarimax = forecast_sarimax.predicted_mean
forecast_ci_sarimax = forecast_sarimax.conf_int()

# Plotting the SARIMAX forecast
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data['Rain_in_past_5_days'], label='Training data', color='blue')
plt.plot(test_data.index, test_data['Rain_in_past_5_days'], label='Actual test data', color='green')
plt.plot(test_data.index, forecast_mean_sarimax, label='SARIMAX forecast', color='red')
plt.fill_between(test_data.index, forecast_ci_sarimax.iloc[:, 0], forecast_ci_sarimax.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMAX Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Rain in Past 5 Days')
plt.legend()
plt.show()

# Evaluate the SARIMAX model
mse_sarimax = mean_squared_error(test_data['Rain_in_past_5_days'], forecast_mean_sarimax)
print(f'Mean Squared Error (MSE) for SARIMAX: {mse_sarimax}')
