# training loop

from data import generate_data
from model import LinearRegressionGD

# Step 1: Generate synthetic business data
# This creates 500 samples with 4 features each
X, y = generate_data()

# Step 2: Initialize the model with appropriate hyperparameters
# Learning rate is small (1e-7) because features have different scales
# (e.g., website_visits ranges 1000-10000, while ad_spend ranges 5-50)
# Without feature scaling, a small learning rate prevents overshooting
model = LinearRegressionGD(lr=1e-7, epochs=2000)

# Step 3: Train the model using gradient descent
# This updates weights and bias to minimize mean squared error
model.fit(X, y)

# Step 4: Display training results
print("Training Complete!")
print("=" * 50)
print(f"Final Loss (MSE): {model.losses[-1]:.4f}")
print(f"Weights: {model.w}")
print(f"Bias: {model.b:.4f}")
print("=" * 50)
