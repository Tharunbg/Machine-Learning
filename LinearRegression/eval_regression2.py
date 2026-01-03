import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

data = load_iris()
X = data.data[:, [2, 3]]  # Petal length, petal width
y = data.data[:, 0:1]  # Sepal length


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.load_weight_bias('regression2_output_model.npz')

mse = model.score(X_test, y_test)
print(f'Mean Squared Error: {mse}')