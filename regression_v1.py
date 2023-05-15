import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import PredictionError, ResidualsPlot

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

# Histograms
df.hist(bins=30, figsize=(10, 10))
plt.tight_layout()
plt.show()

# Correlation Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Creating X and Y
x = df.drop('income', axis=1)
y = df['income']

# Creating Train and Test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model Creation and Training
model = LinearRegression()
model.fit(x_train, y_train)

# Model Evaluation
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_predict))
r2_train = metrics.r2_score(y_train, y_train_predict)

rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_predict))
r2_test = metrics.r2_score(y_test, y_test_predict)

print("Training set stats:")
print('RMSE:', rmse_train)
print('R2 Score:', r2_train)

print("Test set stats:")
print('RMSE:', rmse_test)
print('R2 Score:', r2_test)

# Residuals Plot
visualizer = ResidualsPlot(model)
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof()

# Prediction Error Plot
visualizer = PredictionError(model)
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof()

# Pairplot
sns.pairplot(df, x_vars=['education', 'age', 'gender_Male'], y_vars='income', kind='reg')
plt.show()
