import numpy as np

def generate_data(m=500):
    """
    Generates realistic small business data for revenue prediction.
    
    Features:
    - ad_spend: Advertising expenditure (in thousands)
    - website_visits: Number of website visitors
    - average_product_price: Average price of products (in dollars)
    - location_score: Location quality score (0-100)
    
    Target:
    - monthly_revenue: Monthly revenue (in thousands)
    
    Args:
        m: Number of samples to generate (default: 500)
    
    Returns:
        X: NumPy matrix of shape (m, 4) containing features
        y: NumPy vector of shape (m,) containing target values
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate features with realistic ranges
    ad_spend = np.random.uniform(5, 50, m)  # $5k to $50k
    website_visits = np.random.uniform(1000, 10000, m)  # 1k to 10k visits
    average_product_price = np.random.uniform(20, 200, m)  # $20 to $200
    location_score = np.random.uniform(30, 95, m)  # 30 to 95 score
    
    # Stack features into matrix X
    X = np.column_stack([ad_spend, website_visits, average_product_price, location_score])
    
    # Generate target with robust linear relationship to prevent NaN
    # Clear linear relationship: revenue = ad_spend * 5 + visits * 0.1 + price * 0.2 + location * 0.5 + base
    # Using stronger, more stable coefficients to prevent NaN issues
    true_weights = np.array([5.0, 0.1, 0.2, 0.5])  # Stronger, more stable coefficients
    true_bias = 10.0  # Base revenue
    
    # Calculate true revenue (without noise) - ensure no NaN
    y_true = X @ true_weights + true_bias
    
    # Ensure y_true has no NaN or inf values
    y_true = np.nan_to_num(y_true, nan=10.0, posinf=1000.0, neginf=10.0)
    
    # Add Gaussian noise to make it realistic (smaller noise to maintain stability)
    noise = np.random.normal(0, 3, m)  # Mean 0, std 3 (reduced from 5)
    y = y_true + noise
    
    # Final safety check: ensure no NaN or inf in final output
    y = np.nan_to_num(y, nan=10.0, posinf=1000.0, neginf=10.0)
    
    return X, y
    
