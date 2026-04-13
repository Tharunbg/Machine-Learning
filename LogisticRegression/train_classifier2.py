import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

load_data = load_iris()
X = load_data.data
y = load_data.target
 
X = X[:, 0:2]
y = (y == 1).astype(int)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Visualize decision regions
plot_decision_regions(X_test, y_test, clf=model)
plt.title("Sepal Length/Width - Logistic Regression")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
 
model.save_weight_bias("logisticregression_sepal.npz")