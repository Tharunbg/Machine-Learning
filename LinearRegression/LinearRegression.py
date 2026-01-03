import numpy as np
 
class LinearRegression:
  def __init__(self):
    self.weights = None
    self.bias = None
 
  def fit(self, input_data, target_value, batch_size=32, regularization=0, max_epochs=100, patience=3):
    self.batch_size = batch_size
    self.regularization = regularization
    self.max_epochs = max_epochs
    self.patience = patience
 
    n_samples, n_features = input_data.shape
    self.weights = np.zeros(n_features)
    self.bias = 0
 
    validation_split = int(0.9 * n_samples)
    X_train = input_data[:validation_split]
    y_train = target_value[:validation_split]
    X_val = input_data[validation_split:]
    y_val = target_value[validation_split:]
 
    train_losses_val = []
 
    best_loss = float('inf')
    patience_counter_val = 0
 
    for epoch_i in range(max_epochs):
        # Shuffle data for mini-batches
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
 
        # Batch training
        for i in range(0, X_train_shuffled.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
 
            # Predictions
            y_prediction = np.dot(X_batch, self.weights) + self.bias
            y_prediction = y_prediction.reshape(-1, 1)  # Ensure y_pred has shape (batch_size, 1)
            error = y_prediction - y_batch
 
            # Compute gradients
            dWeight = (2 / batch_size) * np.dot(X_batch.T, error).flatten()  # Flatten to (n_features,)
            dbias = (2 / batch_size) * np.sum(error)  # Scalar value
 
            #adding regularizations
            dWeight += (2 * self.regularization * self.weights)
 
            # Updated weights and bias
            Learning_rate = 0.01
            self.weights -= Learning_rate * dWeight  
            self.bias -= Learning_rate * dbias
 
        # Calculate training loss for calculating (MSE)
        train_loss = np.mean((y_train - self.predict(X_train)) ** 2)
        train_losses_val.append(train_loss)
 
        # finding Validation loss
        val_predictions = self.predict(X_val)
        val_loss_value = np.mean((y_val - val_predictions) ** 2)
 
        # checking early stopping
        if val_loss_value < best_loss:
            best_loss = val_loss_value
            best_Weight = self.weights
            best_bias = self.bias
            patience_counter_val = 0
        else:
            patience_counter_val += 1
            if patience_counter_val >= patience:
                print(f"Early stopping at epoch is : {epoch_i + 1}")
                break
 
    self.weights = best_Weight
    self.bias = best_bias
    return train_losses_val
 
  def predict(self, input_data):
    return np.dot(input_data, self.weights) + self.bias
 
  def score(self, input_data, target_value):
    y_prediction = self.predict(input_data)
    mse_val = np.mean((target_value - y_prediction) ** 2)
    print(mse_val)
    return mse_val
 
  def save_weight_bias(self, filename):
    np.savez(filename, weights=self.weights, bias=self.bias)
 
  def load_weight_bias(self, filename):
    data_load = np.load(filename)
    self.weights = data_load['weights']
    self.bias = data_load['bias']