import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

load_data = load_iris()
X = load_data.data[:, [0, 1]]  # Sepal length, sepal width
y = load_data.data[:, 2:3]  # Petal length

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.load_weight_bias('regression1_output_model.npz')

mse = model.score(X_test, y_test)
print(f'Mean Squared Error: {mse}')
