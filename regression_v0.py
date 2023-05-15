import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Assume we have a DataFrame 'df' with 'income', 'education', 'age', 'gender' columns
# df = pd.read_csv('your_data.csv')

# Creating a hypothetical DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),
    'education': np.random.randint(1, 21, 1000),
    'age': np.random.randint(20, 65, 1000),
    'gender': np.random.choice(['Male', 'Female'], 1000)
})

# Convert categorical variable into dummy/indicator variables.
df = pd.get_dummies(df, drop_first=True)

# Creating X and Y
X = df.drop('income', axis=1)
Y = df['income']

# Creating Train and Test splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Model Creation and Training
model = LinearRegression()
model.fit(X_train, Y_train)

# Model Evaluation
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, y_train_predict))
r2_train = metrics.r2_score(Y_train, y_train_predict)

rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, y_test_predict))
r2_test = metrics.r2_score(Y_test, y_test_predict)

print("Training set stats:")
print('RMSE:', rmse_train)
print('R2 Score:', r2_train)

print("Test set stats:")
print('RMSE:', rmse_test)
print('R2 Score:', r2_test)

# Plotting
sns.pairplot(df, x_vars=['education', 'age', 'gender_Male'], y_vars='income', kind='reg')
plt.show()
