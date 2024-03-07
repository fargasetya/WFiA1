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

df=pd.read_csv(r'C:/Users/reube/Downloads/WFiA1/weatherAUS.csv')

#change date to date time
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df.sort_values(by='Date', inplace=True)

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

# Create and train the RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
threshold = 0.3
y_pred_probs = rf_classifier.predict_proba(X_test)
y_pred_adjusted = (y_pred_probs[:, 1] > threshold).astype(int)

# Define features (X) and target variable (y) for the upsampled dataset
X_upsampled = df_upsampled[['Rain_in_past_5_days']]
y_upsampled = df_upsampled['RainTomorrow']

# Define parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_upsampled, y_upsampled)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Perform RandomizedSearchCV
randomized_search = RandomizedSearchCV(rf_classifier, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
randomized_search.fit(X_upsampled, y_upsampled)

# Get the best parameters
best_params = randomized_search.best_params_
print(f'Best Parameters: {best_params}')