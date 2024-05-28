import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def create_dummies(df, column, sep=None):
    df[column] = df[column].fillna('None')  # Fill None values with 'None'
    if sep is None:
        dummies = df[column].str.get_dummies()
    else:
        dummies = df[column].str.get_dummies(sep=sep)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[column])
    return df

# Load the dataset
df = pd.read_csv("cleaned_data.csv").copy()

# Create dummy variables for categorical columns
for column in ['benefits', 'employee_count', 'experience', 'lang_tool']:
    sep = ', ' if column == 'lang_tool' else None
    df = create_dummies(df, column, sep)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'position' and 'level' columns
for column in ['position', 'level']:
    df.loc[:, column] = label_encoder.fit_transform(df[column])

# Separate independent variables (X) and dependent variable (y)
X = df.drop(columns=['timestamp', 'salary_currency', 'salary'])
y = df['salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the performance of the model
for dataset, predictions in [('Training', model.predict(X_train)), ('Test', model.predict(X_test))]:
    rmse = mean_squared_error(y_train if dataset == 'Training' else y_test, predictions, squared=False)
    print(f"{dataset} set RMSE: {rmse}")