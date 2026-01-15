import numpy as np

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent from scratch.
    
    Implements mean squared error minimization using explicit gradient calculations.
    No external ML libraries used - pure NumPy implementation.
    """
    
    def __init__(self, lr=0.01, epochs=1000):
        """
        Initialize the linear regression model.
        
        Args:
            lr: Learning rate for gradient descent (default: 0.01)
            epochs: Number of training iterations (default: 1000)
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None  # Weights (coefficients)
        self.b = None  # Bias (intercept)
        self.losses = []  # Loss history
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X: Feature matrix of shape (m, n) where m is samples, n is features
            y: Target vector of shape (m,)
        """
        m, n = X.shape
        
        # Initialize weights and bias to small random values
        self.w = np.random.randn(n) * 0.01
        self.b = 0.0
        
        # Gradient descent loop
        for epoch in range(self.epochs):
            # Forward pass: compute predictions
            y_pred = X @ self.w + self.b
            
            # Compute mean squared error
            mse = np.mean((y_pred - y) ** 2)
            self.losses.append(mse)
            
            # Compute gradients explicitly
            # Gradient of MSE w.r.t. weights: (2/m) * X^T * (y_pred - y)
            dw = (2 / m) * X.T @ (y_pred - y)
            
            # Gradient of MSE w.r.t. bias: (2/m) * sum(y_pred - y)
            db = (2 / m) * np.sum(y_pred - y)
            
            # Update weights and bias using gradient descent
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
    
    def predict(self, X):
        """
        Make predictions using learned weights and bias.
        
        Args:
            X: Feature matrix of shape (m, n)
        
        Returns:
            Predictions vector of shape (m,)
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        return X @ self.w + self.b

