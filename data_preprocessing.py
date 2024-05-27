import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Data Read
df = pd.read_csv('software-salary.csv')

# Strip leading and  trailing spaces from column names
df.columns = df.columns.str.strip()

# Date and  Time Format Fix
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S')

# Categorical Data Encoding
df['gender'] = df['gender'].str.capitalize()
df['location'] = df['location'].str.strip()
df['work_arrangement'] = df['work_arrangement'].str.strip()

# Fill Missing Values
df['benefits'].fillna('Unknown', inplace=True)
df['level'].fillna('Unknown', inplace=True)
df['experience'].fillna('Unknown', inplace=True)
df['lang_tool'].fillna('Unknown', inplace=True)
df['salary_currency'].fillna('Unknown', inplace=True)

def convert_salary(salary):
    salary = salary.replace('.', '').replace(',', '.')
    if '-' in salary:
        min_salary, max_salary = salary.split('-')
        average_salary = (float(min_salary.strip()) + float(max_salary.strip())) / 2
    else:
        try:
            average_salary = float(salary)
        except ValueError:
            average_salary = np.nan  # or a default value
    return int(round(average_salary)) if not np.isnan(average_salary) else average_salary

df['salary'] = df['salary'].apply(convert_salary)

# Transforming categorical data with one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'location', 'work_arrangement'])

# Identifying and  filtering outliers
q1 = df['salary'].quantile(0.25)
q3 = df['salary'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]

# Encoding position and  level with label encoding
label_encoder = LabelEncoder()
df['position'] = label_encoder.fit_transform(df['position'])
df['level'] = label_encoder.fit_transform(df['level'])

# Saving cleaned data
df.to_csv('cleaned_data.csv', index=False)
