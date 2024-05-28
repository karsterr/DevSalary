import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("cleaned_data.csv").copy()  # Make a copy of the DataFrame to avoid SettingWithCopyWarning

# Assuming 'benefits' is the column with the categorical data
benefits = df['benefits'].str.get_dummies(sep=', ')
df = pd.concat([df, benefits], axis=1)

# Now you can drop the original 'benefits' column as its information is now represented by the dummy variables
df = df.drop(columns=['benefits'])

# Assuming df is your DataFrame and 'employee_count' is the column with the categorical data
employee_count = df['employee_count'].str.get_dummies()
df = pd.concat([df, employee_count], axis=1)

# Now you can drop the original 'employee_count' column as its information is now represented by the dummy variables
df = df.drop(columns=['employee_count'])

# Assuming df is your DataFrame and 'experience' is the column with the categorical data
experience = df['experience'].str.get_dummies()
df = pd.concat([df, experience], axis=1)

# Now you can drop the original 'experience' column as its information is now represented by the dummy variables
df = df.drop(columns=['experience'])

# Assuming df is your DataFrame and 'lang_tool' is the column with the categorical data
lang_tool = df['lang_tool'].str.get_dummies(sep=', ')
df = pd.concat([df, lang_tool], axis=1)

# Now you can drop the original 'lang_tool' column as its information is now represented by the dummy variables
df = df.drop(columns=['lang_tool'])

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