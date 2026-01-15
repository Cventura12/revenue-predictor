"""
Database operations module.
Separates database logic from API endpoints with comprehensive error handling.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from supabase import Client
import traceback


def save_prediction_to_db(
    supabase_client: Optional[Client],
    ad_spend: float,
    website_visits: float,
    average_product_price: float,
    location_score: float,
    predicted_revenue: float,
    user_id: Optional[str] = None
) -> bool:
    """
    Save prediction to Supabase with comprehensive error handling.
    
    Args:
        supabase_client: Supabase client instance (can be None)
        ad_spend: Advertising spend value
        website_visits: Website visits value
        average_product_price: Average product price
        location_score: Location score
        predicted_revenue: Predicted revenue value
        user_id: User ID to associate with the prediction (required for data security)
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not supabase_client:
        return False
    
    if not user_id:
        print("Warning: user_id is required but not provided. Prediction not saved.")
        return False
    
    try:
        # Ensure all values are floats
        prediction_data = {
            "user_id": user_id,
            "ad_spend": float(ad_spend),
            "website_visits": float(website_visits),
            "average_product_price": float(average_product_price),
            "location_score": float(location_score),
            "predicted_revenue": float(predicted_revenue),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Attempt to insert with timeout handling
        result = supabase_client.table('predictions').insert(prediction_data).execute()
        
        if result and hasattr(result, 'data') and result.data:
            print("Prediction saved to Supabase successfully")
            return True
        else:
            print("Warning: Supabase insert returned no data")
            return False
            
    except Exception as db_error:
        error_msg = str(db_error)
        print(f"Failed to save prediction to Supabase: {error_msg}")
        
        # Handle specific error types
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            print("Database connection issue - prediction still returned to user")
        elif "duplicate" in error_msg.lower():
            print("Duplicate entry detected - skipping save")
        else:
            print(f"Unknown database error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return False


def get_prediction_history(
    supabase_client: Optional[Client],
    user_id: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve prediction history from Supabase with error handling.
    
    Args:
        supabase_client: Supabase client instance (can be None)
        user_id: User ID to filter predictions (required for data security)
        limit: Maximum number of records to return
    
    Returns:
        List[Dict[str, Any]]: List of prediction records, empty list on error
    """
    if not supabase_client:
        return []
    
    if not user_id:
        print("Warning: user_id is required but not provided. Returning empty history.")
        return []
    
    try:
        # Query with error handling - filter by user_id for data security
        query = supabase_client.table('predictions')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .limit(limit)
        
        response = query.execute()
        
        # Validate response
        if not response or not hasattr(response, 'data'):
            print("Warning: Supabase query returned invalid response")
            return []
        
        # Ensure all numeric values are floats
        if response.data:
            for record in response.data:
                for key in ['ad_spend', 'website_visits', 'average_product_price', 
                           'location_score', 'predicted_revenue']:
                    if key in record and record[key] is not None:
                        try:
                            record[key] = float(record[key])
                        except (ValueError, TypeError):
                            record[key] = 0.0
        
        return response.data if response.data else []
        
    except Exception as query_error:
        error_msg = str(query_error)
        print(f"Supabase query error: {error_msg}")
        
        # Handle specific error types
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            print("Database connection timeout - returning empty history")
        else:
            print(f"Traceback: {traceback.format_exc()}")
        
        return []


def get_all_predictions_for_training(
    supabase_client: Optional[Client],
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve ALL prediction records from Supabase for ML training.
    This fetches all historical data, not just a limited set.
    
    Args:
        supabase_client: Supabase client instance (can be None)
        user_id: User ID to filter predictions (required for data security)
    
    Returns:
        List[Dict[str, Any]]: List of all prediction records for the user
    """
    if not supabase_client:
        return []
    
    if not user_id:
        print("Warning: user_id is required but not provided. Returning empty data for training.")
        return []
    
    try:
        # Query ALL records (no limit) - filter by user_id for data security
        query = supabase_client.table('predictions')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)
        
        response = query.execute()
        
        # Validate response
        if not response or not hasattr(response, 'data'):
            print("Warning: Supabase query returned invalid response")
            return []
        
        # Ensure all numeric values are floats
        if response.data:
            for record in response.data:
                for key in ['ad_spend', 'website_visits', 'average_product_price', 
                           'location_score', 'predicted_revenue']:
                    if key in record and record[key] is not None:
                        try:
                            record[key] = float(record[key])
                        except (ValueError, TypeError):
                            record[key] = 0.0
        
        return response.data if response.data else []
        
    except Exception as query_error:
        error_msg = str(query_error)
        print(f"Supabase query error for training data: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return []
