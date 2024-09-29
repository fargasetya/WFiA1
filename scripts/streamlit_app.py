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
    pages = ["Exploration", "Data Visualization", "Modelling"]
    page = st.sidebar.radio("Go to", pages)

    # Load data and model
    df = pd.read_csv('../data/weatherAUS.csv')
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    columns_to_check = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

    if page == pages[0]:
        st.write("## Presentation of data")

        # Display first few lines of df
        st.write("### First Few lines of df")
        st.write(df.head())

        # Display some statistics
        st.write("### Some statistics")
        st.write(df.describe())

        # Display data types
        st.subheader("Data Types")
        st.code(df.dtypes)
        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())

        # Display number of unique values
        st.subheader("Number of Unique Values")
        for col in df.columns:
            st.code(f"{col}: {df[col].nunique()}")

        # Display message about null and extreme values
        st.subheader("Note")
        st.write("Null values and extreme values need to be dealt with before proceeding with further analysis.")         

    elif page == pages[1]:
        st.write("### Data Visualization")
        
        null_counts = df.isnull().sum()
        filtered_null_counts = null_counts[null_counts > 0]
        st.write("#### Seems like there are too many nulls in Evaporation, Sunshine, Cloud9am and Cloud3pm. This data may be valuable for training.")

        plt.figure(figsize=(10, 6))
        filtered_null_counts.plot(kind='bar')
        plt.title('Count of Null Values per Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Null Values')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)

        st.write("#### Checking if of them are normally distributed. If yes, we can replace with mean/median/mode")
        columns_to_check = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

        # Plot histograms for each column
        for column in columns_to_check:
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=30, alpha=0.5, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # Generate QQ-Plots for normality check
        st.write("#### Histograms can be misleading, lets do QQ Plots to check normality")
        for column in columns_to_check:
            fig = qq_plot(df[column].dropna(), column)
            st.pyplot(fig)

        st.write("#### None of these columns are normally distributed. Replacing with mean/median is not recommended.")
        st.write("#### Now, let's check how they are distributed over time based on Location.")

        # Convert 'Date' column to datetime if not already
        groupedDf = df.copy()
        
        # Group the data by Location and Date and calculate the mean for each numeric column
        numeric_columns = groupedDf.select_dtypes(include=[np.number]).columns
        grouped = groupedDf.groupby(['Location', 'Date'])[numeric_columns].mean().reset_index()

        locations = grouped['Location'].unique()

        # Create a figure and axes for the time distribution
        fig, ax = plt.subplots(figsize=(12, 8))

        # Iterate over each location and plot Evaporation, Sunshine, Cloud9am, and Cloud3pm
        for location in locations:
            location_data = grouped[grouped['Location'] == location]
            ax.plot(location_data['Date'], location_data['Evaporation'], label=f'{location} Evaporation')
            ax.plot(location_data['Date'], location_data['Sunshine'], label=f'{location} Sunshine')
            ax.plot(location_data['Date'], location_data['Cloud9am'], label=f'{location} Cloud9am')
            ax.plot(location_data['Date'], location_data['Cloud3pm'], label=f'{location} Cloud3pm')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Distribution by Location and Time')
        ax.legend(ncol=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        st.pyplot(fig)

        st.write("#### Seems like the data until end of 2008 is choppy so we can drop the data before this date. Also seems like for Evaporation and Sunshine, the data is pretty seasonal, so we can replace nulls using interpolation.")

        # Interpolate missing values for seasonal columns
        df.loc[:, ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']] = df[['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']].interpolate(method='linear', limit_direction='forward')

        # Check null values after interpolation
        st.write("#### Checking remaining null values after interpolation for seasonal data")
        null_counts = df.isnull().sum()

        # Filter out columns with zero null values
        filtered_null_counts = null_counts[null_counts > 0]

        if not filtered_null_counts.empty:
            plt.figure(figsize=(10, 6))
            filtered_null_counts.plot(kind='bar')
            plt.title('Count of Null Values per Column')
            plt.xlabel('Columns')
            plt.ylabel('Number of Null Values')
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write("No null values remain after interpolation.")

        # Drop NA values for categorical columns 'RainToday' and 'RainTomorrow'
        st.write("#### We drop null values in 'RainToday' or 'RainTomorrow'")
        df = df.dropna(subset=['RainToday', 'RainTomorrow'])

        # Impute missing values for remaining columns
        st.write("#### We now replace all null based on data types - Mean for Floats and mode for object.")
        for column in df.columns:
            if df[column].dtype == "float64":
                df[column] = df[column].fillna(df[column].mean())
            elif df[column].dtype == "object":
                df[column] = df[column].fillna(df[column].mode()[0])

        # Final null value check
        st.write("#### Checking again is there are any more null values")
        null_counts = df.isnull().sum()

        if null_counts.sum() == 0:
            st.write("No more missing values remain in the dataset.")
        else:
            st.write(null_counts)
        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())

        # Handle extreme values
        st.write("### Handling extreme values")

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

        st.write("#### Boxplots for quantitative variables")
        plot_boxplots(df, quantitative_variables)
        st.pyplot(plt)

        # Handle outlier in 'Evaporation'
        st.write("#### All of the outliers lie within range of explainable extreme weather events except for Evaporation which can't be more than 100.")
        df['Evaporation'] = np.where(df['Evaporation'] > 100, df['Evaporation'].mean(), df['Evaporation'])

        # Plot the correlation matrix
        st.write("#### Correlation matrix to check which features are most important.")
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
        st.write("### Modelling and Predictions")
        
        # Load data
        df = load_data()
        # Convert categorical variables to numerical
        #categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday', 'RainTomorrow']
        #df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
        # Load the SARIMAX model
        sarimax_result = load_model()

        # Display SARIMAX model summary
        st.write("SARIMAX Model Summary")
        sarimax_summary = sarimax_result.summary()
        st.text(sarimax_summary)
        
        features = df.drop(columns=['RainTomorrow_Yes'])
        target = df['RainTomorrow_Yes']

        train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42)

        # Define your features and target variables
        train_predictions = sarimax_result.predict(start=0, end=len(train_target)-1, exog=train_features)
        st.text(train_predictions.describe())
        # Make predictions on the test data
        test_predictions = sarimax_result.predict(start=len(train_target), 
                                          end=len(train_target) + len(test_target) - 1, 
                                          exog=test_features)
        
        # Convert predictions to binary classifications
        test_predictions_binary = np.where(test_predictions > 0.5, 1, 0)
        # Calculate accuracy
        accuracy = accuracy_score(test_target, test_predictions_binary)
        st.write(f"### Accuracy: {accuracy}")

        # Calculate other metrics
        mse = mean_squared_error(test_target, test_predictions)
        mae = mean_absolute_error(test_target, test_predictions)
        r2 = r2_score(test_target, test_predictions)

        st.write(f"### Mean Squared Error (MSE): {mse}")
        st.write(f"### Mean Absolute Error (MAE): {mae}")
        st.write(f"### R^2 Score: {r2}")

        # Display classification report
        st.write("### Classification Report")
        report = classification_report(test_target, test_predictions_binary, output_dict=True)
        st.text(classification_report(test_target, test_predictions_binary))

if __name__ == "__main__":
    main()
