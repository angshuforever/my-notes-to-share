# Data Preprocessing and Feature Selection Guide for Linear Regression

This guide walks through the process of cleaning a large dataset and selecting the most relevant features for linear regression using Python's data science libraries.

## Table of Contents
1. [Initial Data Loading and Exploration](#1-initial-data-loading-and-exploration)
2. [Handling Missing Values](#2-handling-missing-values)
3. [Detecting and Handling Bad Data](#3-detecting-and-handling-bad-data)
4. [Feature Selection](#4-feature-selection)
5. [Final Dataset Preparation](#5-final-dataset-preparation)
6. [Complete Example](#6-complete-example)

## 1. Initial Data Loading and Exploration

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

# Missing values summary
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
print("\nMissing Values Summary:")
print(missing_percentages[missing_percentages > 0])

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())
```

## 2. Handling Missing Values

```python
def handle_missing_values(df, threshold=30):
    """
    Handle missing values in the dataset
    
    Parameters:
    df: pandas DataFrame
    threshold: maximum percentage of missing values allowed (default: 30)
    
    Returns:
    cleaned DataFrame
    """
    # Calculate missing value percentages
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # For remaining columns, impute missing values
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    
    # Numeric imputation
    numeric_imputer = SimpleImputer(strategy='median')
    df_cleaned[numeric_columns] = numeric_imputer.fit_transform(df_cleaned[numeric_columns])
    
    # Categorical imputation
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned[categorical_columns] = categorical_imputer.fit_transform(df_cleaned[categorical_columns])
    
    return df_cleaned

# Apply missing value handling
df_cleaned = handle_missing_values(df)
```

## 3. Detecting and Handling Bad Data

```python
def handle_bad_data(df):
    """
    Handle bad data by removing outliers and invalid values
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    cleaned DataFrame
    """
    df_cleaned = df.copy()
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    
    for column in numeric_columns:
        # Calculate IQR
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with boundaries
        df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
        df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
    
    return df_cleaned

# Apply bad data handling
df_cleaned = handle_bad_data(df_cleaned)
```

## 4. Feature Selection

```python
def select_features(X, y, n_features=10):
    """
    Select the best features for linear regression
    
    Parameters:
    X: feature DataFrame
    y: target variable
    n_features: number of features to select
    
    Returns:
    selected feature names, transformed X
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select best features using f_regression
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Calculate and sort feature importance scores
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    return selected_features, X_selected, scores

# Assuming 'target' is your target variable
X = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']

# Select features
selected_features, X_selected, feature_scores = select_features(X, y)

print("\nTop Selected Features:")
print(feature_scores.head(10))
```

## 5. Final Dataset Preparation

```python
def prepare_final_dataset(df, selected_features, target_column):
    """
    Prepare the final dataset with selected features
    
    Parameters:
    df: original DataFrame
    selected_features: list of selected feature names
    target_column: name of target variable
    
    Returns:
    final DataFrame
    """
    final_columns = selected_features + [target_column]
    final_df = df[final_columns].copy()
    
    return final_df

# Prepare final dataset
final_df = prepare_final_dataset(df_cleaned, selected_features, 'target')
```

## 6. Complete Example

Here's a complete example putting all the steps together:

```python
# Load and process the data
df = pd.read_csv('your_dataset.csv')

# Handle missing values
df_cleaned = handle_missing_values(df)

# Handle bad data
df_cleaned = handle_bad_data(df_cleaned)

# Prepare features and target
X = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']

# Select features
selected_features, X_selected, feature_scores = select_features(X, y)

# Prepare final dataset
final_df = prepare_final_dataset(df_cleaned, selected_features, 'target')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    final_df[selected_features],
    final_df['target'],
    test_size=0.2,
    random_state=42
)

# Train and evaluate model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': abs(model.coef_)
}).sort_values('Coefficient', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.title('Feature Importance in Linear Regression Model')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()
```

This code provides a complete pipeline for:
1. Loading and exploring your data
2. Handling missing values through both deletion and imputation
3. Detecting and handling bad data using statistical methods
4. Selecting the most relevant features using f_regression scores
5. Preparing the final dataset
6. Training and evaluating a linear regression model

You can adjust the parameters in each function (such as the missing value threshold or number of features to select) based on your specific needs.

Next steps:
- Adjust the missing value threshold based on your domain knowledge
- Consider different imputation strategies for different types of features
- Validate the selected features with domain experts
- Check the model's assumptions before finalizing it

The final dataset will contain only the most relevant features and cleaned data, ready for linear regression modeling.
