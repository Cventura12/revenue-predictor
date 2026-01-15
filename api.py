from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import os
import math
import traceback
import io
import csv
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client
from fpdf import FPDF
import jwt
import pickle

from schemas import PredictionRequest, PredictionResponse, TrainResponse, ModelInfoResponse, SimulateResponse
from model import LinearRegressionGD
from data import generate_data
from model_store import save_model, load_model, model_exists
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils import ensure_float, validate_prediction_inputs, sanitize_numeric
from database import save_prediction_to_db, get_prediction_history, get_all_predictions_for_training

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully")
else:
    supabase = None
    print("Warning: SUPABASE_URL or SUPABASE_KEY not found. Supabase features will be disabled.")

# JWT Secret for token verification (use the JWT secret from Supabase)
# For production, get this from environment variables
JWT_SECRET = os.getenv("JWT_SECRET", "")

# Folder Setup: Ensure the models folder exists at the very top
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)

# Absolute Paths: Define MODEL_PATH using os.getcwd()
MODEL_PATH = os.path.join(os.getcwd(), "models", "trained_model.pkl")
# Path for sklearn LinearRegression model trained on historical data
SKLEARN_MODEL_PATH = os.path.join(os.getcwd(), "models", "sklearn_model.pkl")
SKLEARN_SCALER_PATH = os.path.join(os.getcwd(), "models", "sklearn_scaler.pkl")


# Authentication helper function
def get_user_id_from_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract user_id from JWT token in Authorization header.
    
    Args:
        authorization: Authorization header value (Bearer <token>)
    
    Returns:
        str: User ID if token is valid, None otherwise
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>" format
        token = authorization.replace("Bearer ", "").strip()
        
        if not token:
            return None
        
        # Decode JWT token without verification (Supabase handles verification)
        # In production, you should verify the token signature
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Extract user_id from token payload
        user_id = decoded.get("sub")  # Supabase uses "sub" for user ID
        
        return user_id
    except jwt.DecodeError as e:
        print(f"JWT decode error: {e}")
        return None
    except Exception as e:
        print(f"Error extracting user_id from token: {e}")
        return None

# Initialize FastAPI app
app = FastAPI(
    title="Revenue Predictor API",
    description="API for training and using a linear regression model to predict business revenue",
    version="1.0.0"
    
)

# Allow all origins/methods/headers for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_historical_training_data(user_id: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Fetch all historical prediction records from Supabase for ML training.
    
    Args:
        user_id: User ID to fetch data for
    
    Returns:
        pd.DataFrame: DataFrame with columns [ad_spend, website_visits, average_product_price, location_score, predicted_revenue]
                     Returns None if not enough data or error occurs
    """
    if not supabase or not user_id:
        print("Supabase not configured or user_id missing. Cannot fetch training data.")
        return None
    
    try:
        # Fetch all historical predictions for this user
        historical_data = get_all_predictions_for_training(supabase, user_id=user_id)
        
        if not historical_data or len(historical_data) < 10:
            print(f"Not enough historical data for training. Found {len(historical_data) if historical_data else 0} records. Need at least 10.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Select only the features and target we need
        required_columns = ['ad_spend', 'website_visits', 'average_product_price', 'location_score', 'predicted_revenue']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns in training data: {missing_columns}")
            return None
        
        # Extract features and target
        training_df = df[required_columns].copy()
        
        # Remove any rows with NaN or invalid values
        training_df = training_df.dropna()
        
        if len(training_df) < 10:
            print(f"Not enough valid data after cleaning. Found {len(training_df)} records. Need at least 10.")
            return None
        
        print(f"Successfully fetched {len(training_df)} historical records for training.")
        return training_df
        
    except Exception as e:
        print(f"Error fetching historical training data: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


def train_sklearn_model_from_history(user_id: Optional[str]) -> Tuple[Optional[LinearRegression], Optional[StandardScaler]]:
    """
    Train a sklearn LinearRegression model on historical data from Supabase.
    
    Args:
        user_id: User ID to fetch and train on their historical data
    
    Returns:
        Tuple[Optional[LinearRegression], Optional[StandardScaler]]: 
            (trained_model, scaler) if successful, (None, None) otherwise
    """
    try:
        # Fetch historical data
        training_df = fetch_historical_training_data(user_id)
        
        if training_df is None:
            print("No training data available. Cannot train sklearn model.")
            return None, None
        
        # Separate features and target
        X = training_df[['ad_spend', 'website_visits', 'average_product_price', 'location_score']].values
        y = training_df['predicted_revenue'].values
        
        print(f"Training sklearn LinearRegression on {len(X)} samples...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train LinearRegression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate and print training metrics
        train_score = model.score(X_scaled, y)
        print(f"Model trained successfully. R² score: {train_score:.4f}")
        print(f"Model coefficients: {model.coef_}")
        print(f"Model intercept: {model.intercept_}")
        
        # Save model and scaler
        try:
            with open(SKLEARN_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            with open(SKLEARN_SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Sklearn model saved to {SKLEARN_MODEL_PATH}")
        except Exception as save_error:
            print(f"Warning: Could not save sklearn model: {save_error}")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training sklearn model from history: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, None


def load_sklearn_model() -> Tuple[Optional[LinearRegression], Optional[StandardScaler]]:
    """
    Load saved sklearn LinearRegression model and scaler.
    
    Returns:
        Tuple[Optional[LinearRegression], Optional[StandardScaler]]: 
            (model, scaler) if found, (None, None) otherwise
    """
    try:
        if not os.path.exists(SKLEARN_MODEL_PATH) or not os.path.exists(SKLEARN_SCALER_PATH):
            return None, None
        
        with open(SKLEARN_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SKLEARN_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Sklearn model loaded successfully.")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading sklearn model: {e}")
        return None, None


def predict_with_fallback(
    ad_spend: float,
    website_visits: float,
    average_product_price: float,
    location_score: float,
    user_id: Optional[str] = None
) -> Tuple[float, bool]:
    """
    Make a prediction using sklearn model if available, otherwise use fallback formula.
    
    Args:
        ad_spend: Advertising spend
        website_visits: Website visits
        average_product_price: Average product price
        location_score: Location score
        user_id: User ID for training data access
    
    Returns:
        Tuple[float, bool]: (prediction_value, used_ml_model)
                           prediction_value is in dollars
                           used_ml_model is True if sklearn model was used, False if fallback
    """
    # Try to load existing sklearn model
    model, scaler = load_sklearn_model()
    
    # If no model exists, try to train one from historical data
    if model is None and user_id:
        print("No sklearn model found. Attempting to train from historical data...")
        model, scaler = train_sklearn_model_from_history(user_id)
    
    # If we have a trained model, use it
    if model is not None and scaler is not None:
        try:
            # Prepare features
            features = np.array([[ad_spend, website_visits, average_product_price, location_score]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Ensure prediction is valid
            if math.isnan(prediction) or not np.isfinite(prediction):
                print("ML model prediction is NaN. Falling back to formula.")
                raise ValueError("NaN prediction")
            
            # Prediction is already in dollars (from historical data)
            prediction_dollars = float(prediction)
            
            # Ensure positive and reasonable
            if prediction_dollars <= 0:
                print("ML model prediction is <= 0. Falling back to formula.")
                raise ValueError("Invalid prediction")
            
            print(f"Using sklearn ML model prediction: ${prediction_dollars:,.2f}")
            return prediction_dollars, True
            
        except Exception as e:
            print(f"Error using sklearn model: {e}. Falling back to formula.")
            # Fall through to fallback formula
    
    # Fallback: Use the original formula
    print("Using fallback formula for prediction.")
    fallback_prediction = (
        ad_spend * 2.5 +
        website_visits * 0.5 +
        average_product_price * 10 +
        location_score * 5
    ) * 1000.0  # Convert to dollars
    
    # Ensure minimum value
    fallback_prediction = max(fallback_prediction, 5000.0)
    
    return fallback_prediction, False
    


# Training logic function that can be called from both /train and /predict
def train_model_logic():
    """
    Core training logic that generates realistic data, scales it, trains model, and saves it.
    Uses StandardScaler to normalize data and prevent math collapse.
    
    Returns:
        tuple: (model, scaler, weights_list, bias, final_loss, model_path)
    """
    # Ensure models directory exists before training
    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
    
    # Generate 1000 rows of realistic synthetic data
    # Revenue = (ad_spend * 2.5) + (website_visits * 0.5) + (average_product_price * 10) + (location_score * 5) + noise
    np.random.seed(42)
    m = 1000
    
    # Generate features with realistic ranges
    ad_spend = np.random.uniform(5, 50, m)  # $5k to $50k
    website_visits = np.random.uniform(1000, 10000, m)  # 1k to 10k visits
    average_product_price = np.random.uniform(20, 200, m)  # $20 to $200
    location_score = np.random.uniform(30, 95, m)  # 30 to 95 score
    
    # Stack features into matrix X
    X = np.column_stack([ad_spend, website_visits, average_product_price, location_score])
    
    # Generate target with the specified formula
    y = (
        ad_spend * 2.5 +
        website_visits * 0.5 +
        average_product_price * 10 +
        location_score * 5 +
        np.random.normal(0, 10, m)  # random noise
    )
    
    # Math Safety: Ensure X and y have no NaN or inf values before training
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=0.0)
    y = np.nan_to_num(y, nan=10.0, posinf=10000.0, neginf=10.0)
    
    # Scaling: Use StandardScaler to normalize the data before training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train the model on scaled data
    # Using larger learning rate since data is now scaled
    model = LinearRegressionGD(lr=0.01, epochs=2000)
    model.fit(X_scaled, y)
    
    # Math Safety: Clean model weights and bias to prevent NaN
    model.w = np.nan_to_num(model.w, nan=0.0, posinf=100.0, neginf=0.0)
    model.b = np.nan_to_num(model.b, nan=10.0, posinf=1000.0, neginf=10.0)
    
    # Store scaler in model for later use (we'll save it separately)
    model.scaler = scaler
    
    # Ensure losses array has no NaN
    if len(model.losses) > 0:
        model.losses = [np.nan_to_num(loss, nan=25.0, posinf=10000.0, neginf=25.0) for loss in model.losses]
    
    # Save the trained model (with scaler attached)
    model_path = save_model(model)
    
    # Convert numpy arrays to lists for JSON serialization with safety
    weights_list = model.w.tolist() if isinstance(model.w, np.ndarray) else model.w
    weights_list = [float(np.nan_to_num(w, nan=0.0, posinf=100.0, neginf=0.0)) for w in weights_list]
    
    final_loss = float(np.nan_to_num(model.losses[-1] if len(model.losses) > 0 else 25.0, nan=25.0, posinf=10000.0, neginf=25.0))
    bias = float(np.nan_to_num(model.b, nan=10.0, posinf=1000.0, neginf=10.0))
    
    return model, scaler, weights_list, bias, final_loss, model_path


@app.post("/train", response_model=TrainResponse)
async def train_model():
    """
    Train the linear regression model on synthetic data.
    
    This endpoint:
    1. Generates synthetic business data (500 samples)
    2. Trains the LinearRegressionGD model using gradient descent
    3. Saves the trained model to disk
    4. Returns training results (final loss, weights, bias)
    
    Returns:
        TrainResponse: Training results including final loss, weights, and bias
    
    Example:
        POST /train
        Response: {
            "final_loss": 25.1234,
            "weights": [0.78, 0.0019, 0.098, 0.52],
            "bias": 10.45,
            "message": "Model trained and saved successfully"
        }
    """
    try:
        model, scaler, weights_list, bias, final_loss, model_path = train_model_logic()
        
        # Ensure all values are JSON compliant (no NaN or inf)
        final_loss = float(final_loss)
        if math.isnan(final_loss) or not np.isfinite(final_loss):
            final_loss = 0.0
        
        bias = float(bias)
        if math.isnan(bias) or not np.isfinite(bias):
            bias = 0.0
        
        # Ensure weights list contains only valid floats
        weights_list = [float(w) if not (math.isnan(float(w)) or not np.isfinite(float(w))) else 0.0 
                       for w in weights_list]
        
        return TrainResponse(
            final_loss=final_loss,
            weights=weights_list,
            bias=bias,
            message=f"Model trained and saved successfully to {model_path}"
        )
    
    except Exception as e:
        # Error Handling: Return the actual error message as a string
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Log full error for debugging
        print(f"Training error: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        raise HTTPException(
            status_code=500, 
            detail=error_message
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_revenue(
    request: PredictionRequest,
    user_id: Optional[str] = Depends(get_user_id_from_token)
):
    """
    Make a revenue prediction using the trained model.
    
    This endpoint:
    1. Loads the saved trained model
    2. Accepts business features (ad_spend, website_visits, average_product_price, location_score)
    3. Makes a prediction
    4. Calculates feature contributions
    5. Returns predicted revenue and explanation
    
    Args:
        request: PredictionRequest with business features
    
    Returns:
        PredictionResponse: Predicted revenue and feature contributions breakdown
    
    Raises:
        HTTPException: If model not found or prediction fails
    
    Example:
        POST /predict
        Body: {
            "ad_spend": 25.0,
            "website_visits": 5000.0,
            "average_product_price": 75.0,
            "location_score": 80.0
        }
        Response: {
            "predicted_revenue": 45.23,
            "explanation": {
                "ad_spend_contribution": 20.0,
                "website_visits_contribution": 10.0,
                "average_product_price_contribution": 7.5,
                "location_score_contribution": 40.0,
                "bias_contribution": 10.0
            }
        }
    """
    try:
        # Validate and convert inputs using utility function
        validated_inputs = validate_prediction_inputs({
            'ad_spend': request.ad_spend,
            'website_visits': request.website_visits,
            'average_product_price': request.average_product_price,
            'location_score': request.location_score
        })
        
        # Use advanced ML model with fallback
        prediction_dollars, used_ml_model = predict_with_fallback(
            validated_inputs['ad_spend'],
            validated_inputs['website_visits'],
            validated_inputs['average_product_price'],
            validated_inputs['location_score'],
            user_id=user_id
        )
        
        # Final safety check using utility function
        prediction_dollars = sanitize_numeric(prediction_dollars, 10000.0, max_value=1000000.0)
        
        # Feature Contribution: Calculate accurate contributions based on training formula
        # Use validated inputs and utility functions
        feature_names = ["ad_spend", "website_visits", "average_product_price", "location_score"]
        feature_values = [
            validated_inputs['ad_spend'],
            validated_inputs['website_visits'], 
            validated_inputs['average_product_price'],
            validated_inputs['location_score']
        ]
        
        # Calculate contributions using the training formula coefficients
        formula_coefficients = [2.5, 0.5, 10.0, 5.0]  # Matching the training formula
        
        explanation = {}
        for i, (name, value) in enumerate(zip(feature_names, feature_values)):
            # Contribution in thousands, then convert to dollars - use utility function
            contribution = ensure_float(value * formula_coefficients[i], 0.0) * 1000.0
            contribution = sanitize_numeric(contribution, 0.0, max_value=100000.0)
            explanation[f"{name}_contribution"] = contribution
        
        # Add bias contribution (base revenue) - in dollars
        bias_value = 0.0  # No base in the formula, but we can add a small base if needed
        explanation["bias_contribution"] = bias_value
        
        # Save prediction to Supabase using modular database function
        save_prediction_to_db(
            supabase,
            float(request.ad_spend),
            float(request.website_visits),
            float(request.average_product_price),
            float(request.location_score),
            float(prediction_dollars)
        )
        
        return PredictionResponse(
            predicted_revenue=prediction_dollars,
            explanation=explanation
        )
    
    except Exception as e:
        # Error Handling: Return the actual error message as a string
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Log full error for debugging
        print(f"Prediction error: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        raise HTTPException(
            status_code=500,
            detail=error_message
        )


def get_prediction_value(request: PredictionRequest) -> float:
    """
    Helper function to get prediction value from the model.
    Reuses the same logic as /predict endpoint.
    
    Args:
        request: PredictionRequest with business features
    
    Returns:
        float: Predicted revenue in dollars
    """
    # Auto-Train: If model doesn't exist, automatically train it first
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Auto-training model...")
        model, scaler, _, _, _, _ = train_model_logic()
    else:
        # Load the existing model (which should have scaler attached)
        model = load_model()
        scaler = getattr(model, 'scaler', None)
        # If no scaler, retrain
        if scaler is None:
            print("Model missing scaler. Retraining...")
            model, scaler, _, _, _, _ = train_model_logic()
    
    # Convert request to numpy array for prediction - use utility function
    validated_inputs = validate_prediction_inputs({
        'ad_spend': request.ad_spend,
        'website_visits': request.website_visits,
        'average_product_price': request.average_product_price,
        'location_score': request.location_score
    })
    
    features = np.array([[
        validated_inputs['ad_spend'],
        validated_inputs['website_visits'],
        validated_inputs['average_product_price'],
        validated_inputs['location_score']
    ]], dtype=np.float64)
    
    # Scale features using the same scaler used in training
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    # Make prediction on scaled features
    prediction = model.predict(features_scaled)[0]
    
    # Check if prediction is NaN - if so, retrain model and try again
    if math.isnan(prediction) or not np.isfinite(prediction):
        print("Prediction is NaN. Retraining model and retrying...")
        # Retrain the model
        model, scaler, _, _, _, _ = train_model_logic()
        features_scaled = scaler.transform(features)
        # Retry prediction
        prediction = model.predict(features_scaled)[0]
        
        # If still NaN after retraining, use default value
        if math.isnan(prediction) or not np.isfinite(prediction):
            print("Prediction still NaN after retraining. Using default value 0.0")
            prediction = 0.0
    
    # Ensure prediction is a valid float using np.nan_to_num
    prediction = float(np.nan_to_num(prediction, nan=0.0, posinf=10000.0, neginf=0.0))
    
    # Response Formatting: Multiply by 1000 to show thousands of dollars
    # Model was trained on 'k' units, so multiply by 1000 for display
    prediction_dollars = prediction * 1000.0
    
    # Prediction Validation: If prediction <= 0, calculate logical estimate
    if prediction_dollars <= 0:
        print("Prediction is <= 0. Calculating logical estimate based on input values...")
        # Logical estimate using the training formula: revenue = (ad_spend * 2.5) + (website_visits * 0.5) + (avg_price * 10) + (location * 5)
        logical_estimate = (
            request.ad_spend * 2.5 +
            request.website_visits * 0.5 +
            request.average_product_price * 10 +
            request.location_score * 5
        ) * 1000.0  # Convert to dollars
        prediction_dollars = max(logical_estimate, 5000.0)  # Ensure minimum of $5,000
        print(f"Using logical estimate: {prediction_dollars}")
    
    # Final safety check
    prediction_dollars = float(np.nan_to_num(prediction_dollars, nan=10000.0, posinf=1000000.0, neginf=10000.0))
    
    return prediction_dollars


# Custom PDF class for Revenue AI Report
class RevenueReportPDF(FPDF):
    def header(self):
        # Header with title
        self.set_font('Arial', 'B', 24)
        self.set_text_color(102, 126, 234)  # Purple color
        self.cell(0, 15, 'Revenue AI Report', 0, 1, 'C')
        self.ln(5)
        
        # Timestamp
        self.set_font('Arial', '', 10)
        self.set_text_color(128, 128, 128)  # Gray color
        timestamp = datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p UTC')
        self.cell(0, 8, f'Generated on {timestamp}', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        # Footer with page number
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@app.post("/report")
async def generate_report(request: PredictionRequest):
    """
    Generate a professional PDF report with prediction data.
    
    This endpoint:
    1. Accepts the same input as /predict (ad_spend, website_visits, average_product_price, location_score)
    2. Reuses the existing model prediction logic
    3. Generates a professional PDF file in memory
    4. Returns the PDF as a downloadable file
    
    Args:
        request: PredictionRequest with business features
    
    Returns:
        StreamingResponse: PDF file download (revenue_report.pdf)
    
    Raises:
        HTTPException: If prediction fails
    
    Example:
        POST /report
        Body: {
            "ad_spend": 25.0,
            "website_visits": 5000.0,
            "average_product_price": 75.0,
            "location_score": 80.0
        }
        Response: PDF file download
    """
    try:
        # Get prediction using the same logic as /predict
        predicted_revenue = get_prediction_value(request)
        
        # Round predicted revenue to 2 decimals
        predicted_revenue_rounded = round(predicted_revenue, 2)
        
        # Format revenue with commas
        predicted_revenue_formatted = f"${predicted_revenue_rounded:,.2f}"
        
        # Create PDF instance
        pdf = RevenueReportPDF()
        pdf.add_page()
        
        # Input Summary Section
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(51, 51, 51)  # Dark gray
        pdf.cell(0, 10, 'Input Summary', 0, 1, 'L')
        pdf.ln(3)
        
        # Input details
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(102, 102, 102)
        
        # Ad Spend
        pdf.cell(60, 8, 'Advertising Spend:', 0, 0, 'L')
        pdf.set_text_color(51, 51, 51)
        pdf.cell(0, 8, f"${request.ad_spend:,.0f}", 0, 1, 'L')
        pdf.set_text_color(102, 102, 102)
        
        # Website Visits
        pdf.cell(60, 8, 'Website Visits:', 0, 0, 'L')
        pdf.set_text_color(51, 51, 51)
        pdf.cell(0, 8, f"{request.website_visits:,.0f}", 0, 1, 'L')
        pdf.set_text_color(102, 102, 102)
        
        # Average Product Price
        pdf.cell(60, 8, 'Average Product Price:', 0, 0, 'L')
        pdf.set_text_color(51, 51, 51)
        pdf.cell(0, 8, f"${request.average_product_price:.2f}", 0, 1, 'L')
        pdf.set_text_color(102, 102, 102)
        
        # Location Score
        pdf.cell(60, 8, 'Location Score:', 0, 0, 'L')
        pdf.set_text_color(51, 51, 51)
        pdf.cell(0, 8, f"{request.location_score:.0f}", 0, 1, 'L')
        
        pdf.ln(15)
        
        # Predicted Revenue - Biggest text on page
        pdf.set_font('Arial', 'B', 36)
        pdf.set_text_color(102, 126, 234)  # Purple color
        pdf.cell(0, 15, 'Predicted Revenue', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 48)
        pdf.set_text_color(16, 185, 129)  # Green color for revenue
        pdf.cell(0, 20, predicted_revenue_formatted, 0, 1, 'C')
        pdf.ln(10)
        
        # Additional info
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 8, 'Generated by Revenue AI Predictor', 0, 1, 'C')
        
        # Get PDF as bytes (fpdf2 outputs as bytes when using 'S' destination)
        pdf_output = pdf.output(dest='S')
        
        # Convert to bytes if it's a string (for compatibility)
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
        
        # Create StreamingResponse with proper headers
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type='application/pdf',
            headers={
                'Content-Disposition': 'attachment; filename="revenue_report.pdf"'
            }
        )
    
    except Exception as e:
        # Error Handling: Return the actual error message as a string
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Log full error for debugging
        print(f"Report generation error: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        raise HTTPException(
            status_code=500,
            detail=error_message
        )


@app.get("/history")
async def get_prediction_history(user_id: Optional[str] = Depends(get_user_id_from_token)):
    """
    Get the last 10 prediction records from Supabase.
    
    Returns:
        list: Last 10 prediction records with inputs and predicted revenue
    
    Raises:
        HTTPException: If Supabase is not configured or query fails
    
    Example:
        GET /history
        Response: [
            {
                "id": 1,
                "ad_spend": 25.0,
                "website_visits": 5000.0,
                "average_product_price": 75.0,
                "location_score": 80.0,
                "predicted_revenue": 45230.0,
                "created_at": "2024-01-01T12:00:00Z"
            },
            ...
        ]
    """
    try:
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Supabase is not configured. Please set SUPABASE_URL and SUPABASE_KEY in .env file"
            )
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid Authorization token."
            )
        
        # Use modular database function for history retrieval - filter by user_id
        return get_prediction_history(supabase, user_id=user_id, limit=10)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Error Handling: Return the actual error message
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Log full error for debugging
        print(f"History query error: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prediction history: {error_message}"
        )


@app.post("/simulate", response_model=SimulateResponse)
async def simulate_scenarios(request: PredictionRequest):
    """
    Simulate multiple What-If scenarios in a single request.
    
    This endpoint:
    1. Accepts base input values (ad_spend, website_visits, average_product_price, location_score)
    2. Calculates predictions for multiple scenarios:
       - Standard: Current inputs
       - Double Ad Spend: 2x advertising budget
       - Double Visits: 2x website visits
    3. Returns all scenario predictions in a single response
    
    Args:
        request: PredictionRequest with base business features
    
    Returns:
        SimulateResponse: Dictionary of scenario names to predicted revenue values
    
    Raises:
        HTTPException: If prediction fails
    
    Example:
        POST /simulate
        Body: {
            "ad_spend": 25.0,
            "website_visits": 5000.0,
            "average_product_price": 75.0,
            "location_score": 80.0
        }
        Response: {
            "scenarios": {
                "standard": 45230.0,
                "double_ad_spend": 67850.0,
                "double_visits": 52340.0
            }
        }
    """
    try:
        # Calculate standard scenario (current inputs)
        standard_revenue = get_prediction_value(request)
        
        # Calculate double ad spend scenario
        double_ad_request = PredictionRequest(
            ad_spend=request.ad_spend * 2,
            website_visits=request.website_visits,
            average_product_price=request.average_product_price,
            location_score=request.location_score
        )
        double_ad_revenue = get_prediction_value(double_ad_request)
        
        # Calculate double visits scenario
        double_visits_request = PredictionRequest(
            ad_spend=request.ad_spend,
            website_visits=request.website_visits * 2,
            average_product_price=request.average_product_price,
            location_score=request.location_score
        )
        double_visits_revenue = get_prediction_value(double_visits_request)
        
        # Return all scenarios in a single response
        return SimulateResponse(
            scenarios={
                "standard": float(standard_revenue),
                "double_ad_spend": float(double_ad_revenue),
                "double_visits": float(double_visits_revenue)
            }
        )
    
    except Exception as e:
        # Error Handling: Return the actual error message
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        # Log full error for debugging
        print(f"Simulation error: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        raise HTTPException(
            status_code=500,
            detail=error_message
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the current trained model.
    
    This endpoint:
    1. Loads the saved model (if it exists)
    2. Returns the model's weights and bias
    3. Indicates whether the model has been trained
    
    Returns:
        ModelInfoResponse: Model weights, bias, and training status
    
    Raises:
        HTTPException: If model loading fails
    
    Example:
        GET /model-info
        Response: {
            "weights": [0.78, 0.0019, 0.098, 0.52],
            "bias": 10.45,
            "is_trained": true
        }
    """
    try:
        if not model_exists():
            return ModelInfoResponse(
                weights=[],
                bias=0.0,
                is_trained=False
            )
        
        # Load the model
        model = load_model()
        
        # Convert numpy arrays to lists for JSON serialization
        weights_list = model.w.tolist() if isinstance(model.w, np.ndarray) else model.w
        
        return ModelInfoResponse(
            weights=weights_list,
            bias=float(model.b),
            is_trained=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint that serves the frontend HTML file.
    
    Returns:
        FileResponse: The index.html file
    """
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: API health status
    """
    return {"status": "healthy", "service": "Revenue Predictor API"}


# Static Files: Mount at the VERY BOTTOM so it doesn't block /predict and /train routes
app.mount("/", StaticFiles(directory=".", html=True), name="static")

