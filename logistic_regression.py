import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('software-salary.csv')
if 'timestamp' in data.columns:
    data.drop(columns=['timestamp'], inplace=True)

# Function to convert range to numeric value
def convert_range_to_numeric(range_str):
    range_str = range_str.replace(',', '').strip()
    if '-' in range_str:
        low, high = range_str.split('-')
        low, high = float(low), float(high)
        return (low + high) / 2
    else:
        return np.nan

# Apply the conversion function to the salary and employee_count columns
data['salary_numeric'] = data['salary'].apply(convert_range_to_numeric)
data['employee_count_numeric'] = data['employee_count '].apply(convert_range_to_numeric)

# Drop rows with NaN salary_numeric or employee_count_numeric values
data = data.dropna(subset=['salary_numeric', 'employee_count_numeric'])

# Perform one-hot encoding on categorical variables
data_encoded = pd.get_dummies(data, columns=['gender', 'benefits', 'location', 'work_arrangement', 'position', 'level', 'experience', 'lang_tool', 'salary_currency'])

# Define the threshold for binary classification
threshold = data['salary_numeric'].median()

# Create a binary target variable
data_encoded['salary_binary'] = (data_encoded['salary_numeric'] > threshold).astype(int)

# Drop the original salary, salary_numeric, and employee_count columns
data_encoded.drop(columns=['salary', 'salary_numeric', 'employee_count '], inplace=True)

X = data_encoded.drop(columns=['salary_binary'])
y = data_encoded['salary_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

X_test['predicted_salary_binary'] = y_pred
X_test['actual_salary_binary'] = y_test.values
