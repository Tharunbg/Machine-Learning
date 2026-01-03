import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

load_data = load_iris()
X = load_data.data[:, [0, 1]]  # Sepal length, sepal width
y = load_data.data[:, 2:3]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
 
model = LinearRegression()
loss_value = model.fit(X_train, y_train, batch_size=32,regularization=0, max_epochs=100, patience=3)
 
print()
plt.plot(loss_value)
plt.title('Loss vs Steps of Sepal length, sepal width -> Petal length and Calculating error without regularization')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.savefig('train_regression1_loss.png')
plt.show()
 
loss_value = model.fit(X_train, y_train, batch_size=32,regularization=0.2, max_epochs=100, patience=3)
 
print()
plt.plot(loss_value)
plt.title('Loss vs Steps of Sepal length, sepal width -> Petal length and Calculating error with regularization = 0.2')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.savefig('train_regression1_loss.png')
plt.show()
 
model.save_weight_bias('regression1_output_model.npz')