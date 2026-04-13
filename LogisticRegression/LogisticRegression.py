import numpy as np  
 
class LogisticRegression:
    def __init__(self):
        self.Weights = None
        self.bias = None
    
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
 
    def fit(self, input_data: np.ndarray , target_value:np.ndarray , batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
 
        n_samples, n_features = input_data.shape
        self.Weights = np.zeros(n_features)
        self.bias = 0
        best_loss = float('inf')
        patience_counter_val = 0
 
        split_data = int(0.9 * n_samples)
        X_train, X_val = input_data[:split_data], input_data[split_data:]
        y_train, y_val = target_value[:split_data], target_value[split_data:]
 
 
        for epoch_i in range(max_epochs):
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]
 
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
 
                # Predictions
                y_pred = self.sigmoid(np.dot(X_batch, self.Weights) + self.bias)
 
                # Gradients
                error_value = y_pred - y_batch
                dWeight = (1 / batch_size) * np.dot(X_batch.T, error_value) + 2 * regularization * self.Weights
                dbias = (1 / batch_size) * np.sum(error_value)
 
                # Update weights and bias
                Learning_rate = 0.01
                self.Weights -= Learning_rate * dWeight  
                self.bias -= Learning_rate * dbias
 
            # Validation loss
            val_predictions = self.sigmoid(np.dot(X_val, self.Weights) + self.bias)
            val_loss_value = -np.mean(y_val * np.log(val_predictions + 1e-8) + (1 - y_val) * np.log(1 - val_predictions + 1e-8))
 
            # Early stopping
            if val_loss_value < best_loss:
                best_loss = val_loss_value
                best_Weights = self.Weights
                best_bias = self.bias
                patience_counter_val = 0
            else:
                patience_counter_val += 1
                if patience_counter_val >= patience:
                    print(f"Early stopping at epoch {epoch_i + 1}")
                    break
 
        # Set best parameters
        self.Weights = best_Weights
        self.bias = best_bias
 
    def predict(self, input_data: np.ndarray)-> np.ndarray:
        return np.round(self.sigmoid(np.dot(input_data, self.Weights) + self.bias))
 
 
    def score(self, input_data: np.ndarray ,target_value: np.ndarray)-> np.ndarray:
        predictions_val = self.predict(input_data)
        accuracy = np.mean(predictions_val == target_value)
        return accuracy
 
    def save_weight_bias(self, filename):
        np.savez(filename, Weights=self.Weights, bias=self.bias)
 
    def load_weight_bias(self, filename):
        data = np.load(filename)
        print("Available keys:", data.files)  # Print the available keys
        self.Weights = data['Weights']  # Make sure the key matches the one used in save_weight_bias
        self.bias = data['bias']