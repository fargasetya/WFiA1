import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, classification_report
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Load data with caching
@st.cache_data
def load_data():
  """
  Loads, cleans, and preprocesses the weather data.

  Returns:
      pandas.DataFrame: The cleaned and preprocessed weather data.
  """

  # Read the CSV data
  df = pd.read_csv('../data/weatherAUS.csv')
  df.dropna(subset=['Date'], inplace=True)
  df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
  columns_to_check = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

  for column in columns_to_check:
        plt.hist(df[column].dropna(), bins=30, alpha=0.5, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

  import scipy.stats as stats

  def qq_plot(data):
    stats.probplot(data, dist="norm", plot=plt)

  for column in columns_to_check:
    qq_plot(df[column].dropna())
  # Filter data starting from 2009-01-01
  df = df[df['Date'] >= '2009-01-01']
  df.set_index('Date', inplace=True)

  df.loc[:, ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']] = df[['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']].interpolate(method='linear', limit_direction='forward')
  df = df.dropna(subset=['RainToday', 'RainTomorrow'])
  for column in df.columns:
    if df[column].dtype == "float64":
        df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtype == "object":
        df[column] = df[column].fillna(df[column].mode()[0])

  df['Evaporation'] = np.where(df['Evaporation'] > 100, df['Evaporation'].mean(), df['Evaporation'])
  categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday', 'RainTomorrow']
  df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
  return df

# Load the pickle model with caching
@st.cache_resource
def load_model():
    with open('../models/sarimax_model_FINAL.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define a function to create QQ plots
def qq_plot(data, column):
    fig, ax = plt.subplots()
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'QQ-Plot of {column}')
    return fig

# Display the app structure
def main():
    st.title("Weather Prediction Australia")
    st.sidebar.title("Table of contents")
    pages = ["Exploration", "Data Visualization", "Modelling", "Results"]
    page = st.sidebar.radio("Go to", pages)

    # Load data and model
    df = pd.read_csv('../data/weatherAUS.csv')
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    columns_to_check = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

    if page == pages[0]:
        st.header("Exploration")
        
        # Display presentation of data
        st.subheader("Presentation of Data")
        st.write("In this section, we'll explore the structure and characteristics of our weather dataset.")
        
        # Display first few lines of df
        st.subheader("First Few Lines of DataFrame")
        st.dataframe(df.head())
        st.caption("These rows show the initial structure of our dataset.")

        # Display some statistics
        st.subheader("Some Statistics")
        st.write("We'll examine various statistical measures to understand the distribution of values.")
        st.dataframe(df.describe())

        # Display data types
        st.subheader("Data Types")
        st.code(df.dtypes)
        st.caption("This shows the data types for each column in our dataset.")

        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())

        # Display number of unique values
        st.subheader("Number of Unique Values")
        for col in df.columns:
            st.code(f"{col}: {df[col].nunique()}")

        # Display message about null and extreme values
        st.subheader("Note")
        st.write("""
        Important observations:
        1. Null values exist in certain columns and may need imputation.
        2. Extreme values might affect model performance and should be handled appropriately.
        """)
        st.caption("Null and extreme values require careful consideration before proceeding with analysis.")
    elif page == pages[1]:
        st.header("Data Visualization")
        
        # Display presentation of data
        st.subheader("Data Visualization")
        st.write("In this section, we'll explore the distribution of our weather dataset and perform initial preprocessing steps.")
        
        # Display null value counts
        st.subheader("Null Value Counts")
        null_counts = df.isnull().sum()
        filtered_null_counts = null_counts[null_counts > 0]
        st.dataframe(filtered_null_counts)
        st.caption("We observe high null values in Evaporation, Sunshine, Cloud9am, and Cloud3pm. These columns contain valuable information for training our model.")

        # Plot null value counts
        plt.figure(figsize=(10, 6))
        filtered_null_counts.plot(kind='bar')
        plt.title('Count of Null Values per Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Null Values')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)

        # Normality check
        st.subheader("Normality Check")
        st.write("We'll examine whether the suspected seasonal columns are normally distributed.")
        
        # Histograms for each column
        st.subheader("Histograms")
        for column in columns_to_check:
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=30, alpha=0.5, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # QQ-plots for normality check
        st.subheader("QQ Plots")
        st.write("QQ plots provide a more reliable way to check for normality.")
        for column in columns_to_check:
            fig = qq_plot(df[column].dropna(), column)
            st.pyplot(fig)

        st.write("""
        Based on these visualizations, we conclude that none of these columns are normally distributed. Replacing with mean/median is not recommended.
        """)

        # Time-based analysis
        st.subheader("Time-based Distribution")
        st.write("Now, let's check how these variables are distributed over time based on Location.")
        
        groupedDf = df.copy()
        numeric_columns = groupedDf.select_dtypes(include=[np.number]).columns
        grouped = groupedDf.groupby(['Location', 'Date'])[numeric_columns].mean().reset_index()

        locations = grouped['Location'].unique()

        fig, ax = plt.subplots(figsize=(12, 8))

        for location in locations:
            location_data = grouped[grouped['Location'] == location]
            ax.plot(location_data['Date'], location_data['Evaporation'], label=f'{location} Evaporation')
            ax.plot(location_data['Date'], location_data['Sunshine'], label=f'{location} Sunshine')
            ax.plot(location_data['Date'], location_data['Cloud9am'], label=f'{location} Cloud9am')
            ax.plot(location_data['Date'], location_data['Cloud3pm'], label=f'{location} Cloud3pm')

        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Distribution by Location and Time')
        ax.legend(ncol=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        st.pyplot(fig)

        st.write("""
        Observations:
        1. Data until the end of 2008 appears choppy, so we can drop data before this date.
        2. Evaporation and Sunshine show strong seasonality patterns, suggesting we can replace nulls using interpolation.
        """)

        # Interpolate missing values
        df.loc[:, ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']] = df[['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']].interpolate(method='linear', limit_direction='forward')

        # Check remaining null values
        null_counts = df.isnull().sum()
        filtered_null_counts = null_counts[null_counts > 0]

        if not filtered_null_counts.empty:
            plt.figure(figsize=(10, 6))
            filtered_null_counts.plot(kind='bar')
            plt.title('Count of Null Values per Column After Interpolation')
            plt.xlabel('Columns')
            plt.ylabel('Number of Null Values')
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write("No null values remain after interpolation.")

        # Drop NA values for categorical columns
        df = df.dropna(subset=['RainToday', 'RainTomorrow'])

        # Impute missing values
        st.write("Replacing null values based on data types:")
        for column in df.columns:
            if df[column].dtype == "float64":
                df[column] = df[column].fillna(df[column].mean())
            elif df[column].dtype == "object":
                df[column] = df[column].fillna(df[column].mode()[0])

        # Final null value check
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            st.write("No more missing values remain in the dataset.")
        else:
            st.write(null_counts)
        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())

        # Handle extreme values
        st.subheader("Handling Extreme Values")
        st.write("We'll identify and handle outliers in quantitative variables.")
        
        quantitative_variables = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainTomorrow', 'RainToday']

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

        st.write("Boxplots for quantitative variables")
        plot_boxplots(df, quantitative_variables)
        st.pyplot(plt)

        # Handle outlier in 'Evaporation'
        df['Evaporation'] = np.where(df['Evaporation'] > 100, df['Evaporation'].mean(), df['Evaporation'])

        # Plot the correlation matrix
        # Correlation matrix
        st.subheader("Correlation Matrix")
        st.write("We'll calculate the correlation between quantitative variables and RainTomorrow.")
        
        df_corr = df.copy()
        df_corr['RainTomorrow'] = df_corr['RainTomorrow'].map({'Yes': 1, 'No': 0})
        df_corr['RainToday'] = df_corr['RainToday'].map({'Yes': 1, 'No': 0})
        
        df_quantitative = df_corr[quantitative_variables]
        correlation_matrix = df_quantitative.corr()['RainTomorrow'].sort_values(ascending=False)

        st.write("Correlation of quantitative variables with RainTomorrow")
        st.write(correlation_matrix)

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_quantitative.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Quantitative Variables')
        st.pyplot(plt)

    elif page == pages[2]:
        st.header("Modelling")
    
        # Load data
        df = load_data()
        
        # Feature selection and target definition
        features = df.drop(columns=['RainTomorrow_Yes'])
        target = df['RainTomorrow_Yes']

        # Seasonal decomposition of the target variable
        result = seasonal_decompose(target, model='additive', period=12)

        # Plot the decomposition
        plt.figure(figsize=(10, 6))
        plt.title('Seasonal Decomposition of RainTomorrow')
        fig = result.plot()
        st.pyplot(fig)

        # Perform Augmented Dickey-Fuller test to check for stationarity
        # adf_test = adfuller(target)

        st.write('#### ADF Statistic: -18.422713270952457')
        st.write('#### p-value: 0.0')
        
        st.write("""
        #### Interpretation:
        - The series is stationary if the p-value is less than 0.05.
        - We observe a very low p-value (0.0), indicating that the series is likely stationary.
        """)
        st.write("""
        #### Conclusion:
        Based on the Augmented Dickey-Fuller test results, we conclude that the RainTomorrow series is stationary. This is crucial for building reliable time series models.
        """)

        st.write("Checking Autocorrelation")
        
        # First-order difference
        target_1 = target.diff().dropna()
        
        # Second-order difference (12 periods)
        target_2 = target_1.diff(periods=12).dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

        plot_acf(target_2, lags=36, ax=ax1)
        plot_pacf(target_2, lags=36, ax=ax2)
        st.pyplot(fig)

        st.write("""
        #### Interpretation of Autocorrelation Plot:
        - The autocorrelation function (ACF) shows the correlation between the series and lagged versions of itself.
        - The partial autocorrelation function (PACF) helps identify the order of differencing required.
        """)
        
        # Load the SARIMAX model    
        sarimax_result = load_model()

        # Display SARIMAX model summary
        st.write("SARIMAX Model Summary")
        sarimax_summary = sarimax_result.summary()
        st.text(sarimax_summary)

        st.write("""
        #### Interpreting the SARIMAX Model Summary:
        - Order: p, d, q parameters represent the autoregressive, differencing, and moving average components.
        - Seasonal parameters (P, D, Q) indicate the seasonal ARIMA model used.
        - AIC and BIC values measure the relative quality of the model.
        - The significance of coefficients indicates the importance of each term in the model.
        """)
        
        st.write("""
        #### Next Steps:
        1. Evaluate the model's performance metrics (e.g., RMSE, MAPE).
        2. Compare the SARIMAX model with other time series models like ARIMA or Prophet.
        3. Cross-validate the model using techniques like walk-forward optimization.
        4. Visualize the residuals to check for remaining patterns or outliers.
        """)
    elif page == pages[3]:
        st.header("Results")
        
        # Load data and prepare features/target
        df = load_data()
        features = df.drop(columns=['RainTomorrow_Yes'])
        target = df['RainTomorrow_Yes']

        # Seasonal decomposition of the target variable
        result = seasonal_decompose(target, model='additive', period=12)

        # Load the SARIMAX model
        sarimax_result = load_model()

        # Display SARIMAX model summary
        st.write("SARIMAX Model Summary")
        sarimax_summary = sarimax_result.summary()
        st.text(sarimax_summary)

        # Split data into training and testing sets
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Make predictions on training data
        train_predictions = sarimax_result.predict(start=0, end=len(train_target)-1, exog=train_features)
        
        # Display training set predictions statistics
        st.write("#### Training Set Predictions Statistics")
        st.dataframe(train_predictions.describe())

        # Make predictions on testing data
        test_predictions = sarimax_result.predict(start=len(train_target), end=len(train_target)+len(test_target)-1, exog=test_features)
        
        # Convert predictions to binary classifications
        test_predictions_binary = np.where(test_predictions > 0.5, 1, 0)
        
        # Calculate performance metrics
        accuracy = accuracy_score(test_target, test_predictions_binary)
        mse = mean_squared_error(test_target, test_predictions)
        mae = mean_absolute_error(test_target, test_predictions)
        r2 = r2_score(test_target, test_predictions)

        st.write(f"#### Model Performance Metrics:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R^2 Score: {r2:.2f}")

        # Display classification report
        st.write("#### Classification Report")
        report = classification_report(test_target, test_predictions_binary, output_dict=True)
        st.text(classification_report(test_target, test_predictions_binary))

        # Plot predicted vs. actual values for specific locations
        train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Make predictions on test data
        test_predictions = sarimax_result.predict(start=len(train_target), end=len(train_target)+len(test_target)-1, exog=test_features)
        
        # Convert predictions to binary classifications
        test_predictions_binary = np.where(test_predictions > 0.3, 1, 0)

        # Create a new DataFrame with the added column
        test_features_new = test_features.copy()
        test_features_new['test_predictions_binary_new'] = test_predictions_binary
        test_features_new['RainTomorrow_Yes'] = test_target

        # Set start and end dates
        start_date = pd.to_datetime('2017-04-01')
        end_date = pd.to_datetime('2017-07-15')

        # Get the first 5 location columns
        location_columns = test_features_new.columns[test_features_new.columns.str.startswith('Location_')][:5]
        
        # Iterate over location columns and plot
        for location in location_columns:
            # Filter based on location and date range
            filtered_data = test_features_new[test_features_new[location] == 1]
            filtered_data = filtered_data[(filtered_data.index >= start_date) & (filtered_data.index <= end_date)]

            # Check if there's data for the location
            if not filtered_data.empty:
                # Extract relevant columns
                plot_data = filtered_data[['test_predictions_binary_new', 'RainTomorrow_Yes']]

                # Plot the data as dots with transparency
                plt.figure(figsize=(12, 6))
                plt.scatter(plot_data.index, plot_data['RainTomorrow_Yes'], label='Actual', alpha=0.5)
                plt.scatter(plot_data.index, plot_data['test_predictions_binary_new'], label='Predicted', alpha=0.5)
                plt.title(f"Predicted vs. Actual Rain_Tomorrow for Location_{location}")
                plt.xlabel("Date")
                plt.ylabel("Rain_Tomorrow")
                plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
                plt.xlim(start_date, end_date)  # Set x-axis limits
                plt.legend()
                st.pyplot(plt.gcf())  # Use st.pyplot to display the plot in Streamlit
            else:
                st.write(f"No data found for Location_{location}")

        st.write("""
        #### Interpretation of Results:
        - The accuracy score indicates the proportion of correctly predicted instances.
        - MSE measures the average squared difference between observed and predicted values.
        - MAE provides the average absolute difference between observed and predicted values.
        - R^2 score ranges from 0 to 1, with higher values indicating better fit.
        """)
        
        st.write("""
        #### Next Steps:
        1. Analyze residuals to check for patterns or outliers.
        2. Compare the SARIMAX model with other time series models (e.g., ARIMA, Prophet).
        3. Perform cross-validation to assess model robustness.
        4. Explore feature importance to identify crucial predictors.
        5. Consider ensemble methods to potentially improve model performance.
        """)
        
if __name__ == "__main__":
    main()
