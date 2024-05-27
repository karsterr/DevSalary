import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("cleaned_data.csv").copy()  # Make a copy of the DataFrame to avoid SettingWithCopyWarning

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode 'position' and 'level' columns using .loc to avoid SettingWithCopyWarning
df.loc[:, 'position'] = label_encoder.fit_transform(df['position'])
df.loc[:, 'level'] = label_encoder.fit_transform(df['level'])

# Separate independent variables (X) and dependent variable (y)
X = df.drop(columns=['timestamp', 'salary_currency', 'salary'])  # Independent variables
y = df['salary']  # Dependent variable (salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training dataset
train_predictions = model.predict(X_train)

# Make predictions on the test dataset
test_predictions = model.predict(X_test)

# Evaluate the performance of the model
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

print("Training set RMSE:", train_rmse)
print("Test set RMSE:", test_rmse)
