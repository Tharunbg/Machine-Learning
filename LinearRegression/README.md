# Linear Regression (From Scratch) — Iris Dataset

This folder contains a **from-scratch Linear Regression implementation in NumPy** along with multiple training and evaluation scripts using the **scikit-learn Iris dataset**.

The workflow trains models, saves learned parameters (**weights and bias**) into `.npz` files, and reloads them for evaluation.

---

## Files in this folder

- `LinearRegression.py`  
  Core Linear Regression class implemented from scratch using NumPy.  
  Supports mini-batch gradient descent, optional L2 regularization, and patience-based early stopping.

- `train_regression1.py` `train_regression2.py` `train_regression3.py` `train_regression4.py`  
  Training scripts using different feature–target combinations from the Iris dataset.  
  Each script trains a model and saves parameters to a `.npz` file.

- `eval_regression1.py` `eval_regression2.py` `eval_regression3.py` `eval_regression4.py`  
  Evaluation scripts that load saved model parameters and compute **Mean Squared Error (MSE)** on test data.

- `regression1_output_model.npz` `regression2_output_model.npz` `regression3_output_model.npz` `regression4_output_model.npz`  
  Saved model weights and bias produced after training.

- `LRreport.pdf`  
  Detailed report documenting the implementation and results.

---

## Dataset

This project uses the **Iris dataset** from `sklearn.datasets.load_iris`.

Feature–target mappings explored:
- Inputs: `[sepal length, sepal width]` → Target: `petal length`
- Inputs: `[petal length, petal width]` → Target: `sepal length`
- Inputs: `[sepal length, sepal width]` → Target: `petal width`
- Inputs: `[petal length, petal width]` → Target: `sepal width`

Each training/evaluation pair explores a different linear relationship within the dataset.

---

## How it works (high level)

1. Load the Iris dataset
2. Select two input features (X) and one target variable (y)
3. Train the model using mini-batch gradient descent
4. Save learned weights and bias to a `.npz` file
5. Load the saved model and evaluate using **Mean Squared Error (MSE)**

---

## Requirements

Install dependencies:

```bash
pip install numpy scikit-learn
