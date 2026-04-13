import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

load_data = load_iris()
X = load_data.data
y = load_data.target
 
X = X[:, 2:4]
y = (y == 1).astype(int)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LogisticRegression()
model.load_weight_bias("logisticregression_petal.npz")
 
accuracy = model.score(X_test, y_test)
print(f"Accuracy is: {accuracy}")

