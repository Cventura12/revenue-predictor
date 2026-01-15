# CLI prediction

from data import generate_data
from model import LinearRegressionGD
import numpy as np

# Step 1: Train the model on synthetic data
print("Training model on synthetic business data...")
X, y = generate_data()
model = LinearRegressionGD(lr=1e-7, epochs=2000)
model.fit(X, y)
print("Model training complete!\n")

# Step 2: Hardcoded example input for prediction
# Example business: Medium ad spend, good website traffic, mid-range prices, good location
example_input = np.array([[25.0, 5000.0, 75.0, 80.0]])  # [ad_spend, website_visits, avg_price, location_score]

# Step 3: Make prediction
predicted_revenue = model.predict(example_input)[0]

# Step 4: Format output for non-technical user
print("=" * 60)
print("REVENUE PREDICTION")
print("=" * 60)
print(f"Business Profile:")
print(f"  • Advertising Spend: ${example_input[0][0]:,.0f},000")
print(f"  • Website Visits: {example_input[0][1]:,.0f}")
print(f"  • Average Product Price: ${example_input[0][2]:,.2f}")
print(f"  • Location Score: {example_input[0][3]:.0f}/100")
print()
print(f"Predicted Monthly Revenue: ${predicted_revenue:,.2f},000")
print("=" * 60)
