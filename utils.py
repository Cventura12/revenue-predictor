"""
Utility functions for data validation and type conversion.
Separates business logic from API endpoints.
"""

import numpy as np
import math
from typing import Dict, Any


def ensure_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        float: Converted float value
    """
    try:
        result = float(value)
        if math.isnan(result) or not np.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def validate_prediction_inputs(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Validate and convert prediction inputs to floats.
    
    Args:
        data: Dictionary with prediction inputs
    
    Returns:
        Dict[str, float]: Validated inputs as floats
    """
    return {
        'ad_spend': ensure_float(data.get('ad_spend', 0.0)),
        'website_visits': ensure_float(data.get('website_visits', 0.0)),
        'average_product_price': ensure_float(data.get('average_product_price', 0.0)),
        'location_score': ensure_float(data.get('location_score', 0.0))
    }


def sanitize_numeric(value: Any, default: float = 0.0, 
                    max_value: float = None, min_value: float = None) -> float:
    """
    Sanitize numeric values for JSON compliance.
    
    Args:
        value: Value to sanitize
        default: Default if invalid
        max_value: Maximum allowed value
        min_value: Minimum allowed value
    
    Returns:
        float: Sanitized float value
    """
    result = ensure_float(value, default)
    
    if max_value is not None and result > max_value:
        result = max_value
    if min_value is not None and result < min_value:
        result = min_value
    
    return result

