# %% [markdown]
# # Import relevant libraries

# %%
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pickle

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %% [markdown]
# # Load the dataset and explore

# %%
df = pd.read_csv('..\\data\\weatherAUS.csv')
#print(df.head())
print(df.describe())
df.info()

# %% [markdown]
# Data cleaning. Checking for missing values and deciding how to handle them. Also converting Date to datetime format and make it index.

# %%
df.dropna(subset=['Date'], inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")


null_counts = df.isnull().sum()

# Filter out columns with zero null values
filtered_null_counts = null_counts[null_counts > 0]

plt.figure(figsize=(10, 6))
filtered_null_counts.plot(kind='bar')
plt.title('Count of Null Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %% [markdown]
# Seems like there are too many nulls in Evaporation, Sunshine, Cloud9am and Cloud3pm.
# All the above may be valuable for prediction. Lets check if they are normally distributed.

# %%
columns_to_check = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

for column in columns_to_check:
    plt.hist(df[column].dropna(), bins=30, alpha=0.5, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# %%
import scipy.stats as stats

def qq_plot(data):
    stats.probplot(data, dist="norm", plot=plt)
    plt.show()

for column in columns_to_check:
    qq_plot(df[column].dropna())


# %%
print(df.columns)

# %% [markdown]
# Seems like none of the columns Evaporation, Sunshine, Cloud9am, or Cloud3pm are normally distributed. Therefore, replacing with mean/median is not recommended.
# Now to check how they are distributed in time.

# %%
# Convert 'Date' column to datetime if it's not already
groupedDf = df.copy()
groupedDf['Date'] = pd.to_datetime(groupedDf['Date'], format='%d/%m/%Y')
groupedDf.set_index('Date', inplace=True)

numeric_columns = groupedDf.select_dtypes(include=[np.number]).columns
grouped = groupedDf.groupby(['Location', 'Date'])[numeric_columns].mean().reset_index()

locations = grouped['Location'].unique()

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Iterate over each location
for location in locations:
    # Filter data for the current location
    location_data = grouped[grouped['Location'] == location]
    
    # Plot each variable for the current location
    ax.plot(location_data['Date'], location_data['Evaporation'], label=f'{location} Evaporation')
    ax.plot(location_data['Date'], location_data['Sunshine'], label=f'{location} Sunshine')
    ax.plot(location_data['Date'], location_data['Cloud9am'], label=f'{location} Cloud9am')
    ax.plot(location_data['Date'], location_data['Cloud3pm'], label=f'{location} Cloud3pm')

# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Distribution by Location and Time')
ax.legend(ncol=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Show the plot
plt.show()




# %% [markdown]
# Seems like the data until end of 2008 is pretty choppy so we drop the data before this date.
# Also seems like for Evaporation and Sunshine, the data is pretty seasonal, so we can replace nulls using interpolation.

# %%
df = df[df['Date'] >= '2009-01-01']
df.set_index('Date', inplace=True)

df.loc[:, ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']] = df[['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']].interpolate(method='linear', limit_direction='forward')

# %% [markdown]
# Lets check nulls again

# %%
null_counts = df.isnull().sum()

# Filter out columns with zero null values
filtered_null_counts = null_counts[null_counts > 0]

plt.figure(figsize=(10, 6))
filtered_null_counts.plot(kind='bar')
plt.title('Count of Null Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# Drop any N/A values in RainToday or RainTomorrow as they are categorical, very relavant to what we are trying to predict and that total number of missing is small in comparison to the dataset.
# 
# For missing values in Windgust directions, we will use the mode as that seems logical and shouldn't affect our final

# %%
# Check for missing values
print(df.isnull().sum())

df = df.dropna(subset=['RainToday', 'RainTomorrow'])

# Impute missing values. For numerical we will use means to replace and for categorical we will use mode.
for column in df.columns:
    if df[column].dtype == "float64":
        df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])

df.describe()

# %% [markdown]
# Let's check nulls again. We should have no more nulls.

# %%
null_counts = print(df.isnull().sum())


# %% [markdown]
# So there are no more Nulls. Now lets deal with extreme values.

# %%
quantitative_variables = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

def plot_boxplots(df, variables, ncols=3):
    """
    Function to plot box plots for a list of variables in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - variables: list of column names to plot
    - ncols: number of columns in the grid of plots
    """
    nrows = len(variables) // ncols + (len(variables) % ncols > 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
    
    for i, var in enumerate(variables):
        row = i // ncols
        col = i % ncols
        sns.boxplot(x=df[var], ax=axes[row, col])
        axes[row, col].set_title(var)
    
    # Remove empty subplots
    for j in range(len(variables), nrows*ncols):
        row = j // ncols
        col = j % ncols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()

plot_boxplots(df, quantitative_variables)

# %% [markdown]
# Looking at the box plots-
# - for Temperatures, -mintemp. Range seems to be -10 to ~35 degrees. Maxtemp between -10 and 50 degress. Both ranges are possible in Australia so these can't be ruled as outliers
# - For Rainfall, value is in mm. The is same rainfall above 350mm. Again this seems reasonable
# - For evaporation, there is only one value which above 100. Assuming this to be an error, let's replace this value with the mean value.
# - windgust and wind speeds all seem to be fine be fine with no outliers
# - Humidity is between 0 and 100, again seems fine.
# - Pressures also seem fine.
# 
# There we just need to replace one value for evaporation, wehre there is one value which above 100. Assuming this to be an error, let's replace this value with the mean value.

# %%
# Replace all values in 'Evaporation' above 100 with the mean
df['Evaporation'] = np.where(df['Evaporation'] > 100, df['Evaporation'].mean(), df['Evaporation'])

# %% [markdown]
# Let's do a correlation matrix.
# For RainToday and RainTomorrow, we can just replace Yes with 1 and No with 0. 
# The remaining qualitative variables can be one-hot encoded later.

# %%
quantitative_variables = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainTomorrow','RainToday']
df_corr = df.copy()
df_corr['RainTomorrow'] = df_corr['RainTomorrow'].map({'Yes': 1, 'No': 0})
df_corr['RainToday'] = df_corr['RainToday'].map({'Yes': 1, 'No': 0})
df_quantitative = df_corr[quantitative_variables]

correlation_matrix = df_quantitative.corr()['RainTomorrow'].sort_values(ascending=False)

#print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(df_quantitative.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Quantitative Variables')
plt.show()

# %% [markdown]
# Now lets on-hot encode the categorical variables.
# First lets check the unique value distribution in all of them.

# %%
# Overview of unique values of categorical columns
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

# Count unique values for each categorical column
for col in categorical_cols:
  n_unique = df[col].nunique()
  print(f"Column '{col}': {n_unique} unique values")

# %%
# Overview of unique values of categorical columns
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

# Count unique values for each categorical column
for col in categorical_cols:
  n_unique = df[col].nunique()
  print(f"Column '{col}': {n_unique} unique values")

# %%
# Convert categorical variables to numerical
categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday', 'RainTomorrow']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

# %% [markdown]
# Now we will scale all the variable for ease of model training.

# %%
df.head()

# %%
# Define features and target
data = df.drop(['RainTomorrow_Yes'], axis=1)
y = df['RainTomorrow_Yes']

#standardizign the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Create a new dataframe with the scaled data, keeping the same column names
scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# combining features and targets
scaled_df = pd.concat([scaled_data, y], axis=1)

scaled_df.head()

# %%
features = df.drop(columns=['RainTomorrow_Yes'])
target = df['RainTomorrow_Yes']

# %%
target.head()

# %%
# Seasonal decomposition of the target variable
result = seasonal_decompose(target, model='additive', period=12)

# Plot the decomposition
result.plot()
plt.show()


# %%
# Perform Augmented Dickey-Fuller test to check for stationarity
adf_test = adfuller(target)

print('ADF Statistic:', adf_test[0])
print('p-value:', adf_test[1])
#for key, value in adf_test[4].items():
#    print('Critical Values:')
#    print(f'   {key}, {value}')

if adf_test[1] <= 0.05:
    print("The series is stationary")
else:
    print("The series is not stationary")

# %% [markdown]
# 

# %%
# Split the data into training and test sets (80% training, 20% test)
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42)

# %%
# Define the SARIMAX model
order = (1, 0, 0)  # (p, d, q)
seasonal_order = (0, 1, 1, 12)  # (P, D, Q, s)

# %%
# Fit the SARIMAX model on the training data
model = sm.tsa.statespace.SARIMAX(train_target, 
                                  exog=train_features, 
                                  order=order, 
                                  seasonal_order=seasonal_order)

# %%
sarimax_result = model.fit()

# %%
# Save the model to a pickle file
with open('sarimax_model_FINAL.pkl', 'wb') as f:
    pickle.dump(sarimax_result, f)

# %%
sarimax_summary = sarimax_result.summary()

# %%
print(sarimax_summary)

# %%
train_predictions = sarimax_result.predict(start=0, end=len(train_target)-1, exog=train_features)

# %%
train_predictions.describe()

# %%
# Make predictions on the test data
test_predictions = sarimax_result.predict(start=len(train_target), 
                                          end=len(train_target) + len(test_target) - 1, 
                                          exog=test_features)

# %%
# Convert predictions to binary classifications
test_predictions_binary = np.where(test_predictions > 0.5, 1, 0)

# %%
accuracy = accuracy_score(test_target, test_predictions_binary)
print(f'Accuracy: {accuracy}')

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate accuracy metrics
mse = mean_squared_error(test_target, test_predictions)
mae = mean_absolute_error(test_target, test_predictions)
r2 = r2_score(test_target, test_predictions)

# Print the accuracy metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')

# %%
# Print the classification report
print('\nClassification Report:')
print(classification_report(test_target, test_predictions_binary))


