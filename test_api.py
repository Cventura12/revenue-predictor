"""
Simple test script to verify the API is working.
Run this to test if the server is accessible.
"""
import requests
import json

BASE_URL = "https://revenue-predictor.onrender.com"

def test_health():
    """Test the root endpoint (returns API version info)"""
    try:
        # Use root endpoint "/" instead of "/health" and add timeout for Render's cold start
        response = requests.get(f"{BASE_URL}/", timeout=30)
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_train():
    """Test the train endpoint"""
    try:
        # Add timeout for Render's cold start
        response = requests.post(f"{BASE_URL}/train", timeout=30)
        print(f"Train endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Final loss: {data['final_loss']:.4f}")
            print(f"Weights: {data['weights']}")
            print(f"Bias: {data['bias']:.4f}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Train failed: {e}")
        return False

def test_predict():
    """Test the predict endpoint"""
    try:
        # Payload matches PredictionRequest schema in schemas.py
        payload = {
            "ad_spend": 25.0,
            "website_visits": 5000.0,
            "average_product_price": 75.0,
            "location_score": 80.0
        }
        # Add timeout for Render's cold start
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        print(f"Predict endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Predicted revenue: ${data['predicted_revenue']:.2f},000")
            print(f"Explanation: {json.dumps(data['explanation'], indent=2)}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Predict failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Revenue Predictor API...")
    print("=" * 50)
    
    if not test_health():
        print("\n❌ Server is not running or not accessible!")
        print("This may be due to Render's cold start. Please wait a moment and try again.")
        exit(1)
    
    print("\n✅ Server is running!")
    print("\n" + "=" * 50)
    
    print("\n1. Testing /train endpoint...")
    test_train()
    
    print("\n2. Testing /predict endpoint...")
    test_predict()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

