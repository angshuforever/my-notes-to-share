# House Prices Dataset Analysis Guide

This guide demonstrates how to download and analyze the House Prices dataset from Hugging Face, clean the data, and build a linear regression model.

## 1. Setting Up and Downloading the Dataset

First, install the required packages:

```bash
pip install datasets pandas numpy scikit-learn seaborn matplotlib
```

Now, let's download and prepare the dataset:

```python
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Hugging Face
dataset = load_dataset("fschroef/house-prices")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset['train'])
print("Dataset Shape:", df.shape)
```

## 2. Initial Data Exploration

```python
# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display missing values summary
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
print("\nColumns with Missing Values:")
print(missing_percentages[missing_percentages > 0])

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Visualize correlation with sale price
plt.figure(figsize=(12, 6))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
plt.bar(range(len(correlation)), correlation)
plt.xticks(range(len(correlation)), correlation.index, rotation=90)
plt.title('Correlation with Sale Price')
plt.tight_layout()
plt.show()
```

## 3. Data Preprocessing

```python
def preprocess_house_data(df):
    """
    Preprocess the house prices dataset
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    # Handle categorical variables
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    
    for column in categorical_columns:
        # Fill missing values with mode
        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
        # Encode categorical variables
        df_cleaned[column] = label_encoder.fit_transform(df_cleaned[column])
    
    # Handle numeric variables
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df_cleaned[numeric_columns] = numeric_imputer.fit_transform(df_cleaned[numeric_columns])
    
    # Remove outliers from SalePrice
    Q1 = df_cleaned['SalePrice'].quantile(0.25)
    Q3 = df_cleaned['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned[
        (df_cleaned['SalePrice'] >= Q1 - 1.5 * IQR) & 
        (df_cleaned['SalePrice'] <= Q3 + 1.5 * IQR)
    ]
    
    return df_cleaned

# Apply preprocessing
df_cleaned = preprocess_house_data(df)
```

## 4. Feature Selection

```python
def select_best_features(df, target_col='SalePrice', n_features=15):
    """
    Select the best features for predicting house prices
    
    Parameters:
    df: pandas DataFrame
    target_col: target variable column name
    n_features: number of features to select
    
    Returns:
    selected feature names, feature scores
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select best features
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    selected_features = feature_scores['Feature'].head(n_features).tolist()
    
    return selected_features, feature_scores

# Select features
selected_features, feature_scores = select_best_features(df_cleaned)
print("\nTop 15 Most Important Features:")
print(feature_scores.head(15))
```

## 5. Model Training and Evaluation

```python
# Prepare final dataset
X = df_cleaned[selected_features]
y = df_cleaned['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\nModel Performance:")
print(f"Train R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': abs(model.coef_)
}).sort_values('Coefficient', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.title('Feature Importance in House Price Prediction')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
y_pred = model.predict(X_test_scaled)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual House Prices')
plt.tight_layout()
plt.show()
```

## 6. Making Predictions

```python
def predict_house_price(model, scaler, features, feature_values):
    """
    Predict house price for new data
    
    Parameters:
    model: trained LinearRegression model
    scaler: fitted StandardScaler
    features: list of feature names
    feature_values: dictionary of feature values
    
    Returns:
    predicted price
    """
    # Create input array
    X_new = pd.DataFrame([feature_values])[features]
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    predicted_price = model.predict(X_new_scaled)[0]
    
    return predicted_price

# Example usage:
sample_house = {
    'OverallQual': 7,
    'GrLivArea': 2000,
    'GarageCars': 2,
    # ... add values for other selected features
}

# predicted_price = predict_house_price(model, scaler, selected_features, sample_house)
# print(f"Predicted house price: ${predicted_price:,.2f}")
```

## Key Insights from the House Prices Dataset

1. The dataset contains various features related to houses, including:
   - Physical properties (square footage, number of rooms)
   - Quality ratings
   - Location information
   - Sale conditions

2. Most important features typically include:
   - Overall Quality
   - Ground Living Area
   - Garage Size
   - Total Basement Square Footage
   - Year Built

3. The model's performance suggests that house prices can be predicted with reasonable accuracy using linear regression, though there might be some non-linear relationships that could be captured with more advanced models.

## Tips for Using This Code

1. Adjust the number of features in `select_best_features()` based on your needs
2. Consider removing more or fewer outliers based on domain knowledge
3. Experiment with different feature selection methods
4. Consider adding polynomial features if you notice non-linear relationships
5. Try other regression models like Random Forest or XGBoost for comparison

Remember to monitor for overfitting and validate your model's assumptions before using it for real predictions.
