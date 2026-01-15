from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    """
    Request schema for revenue prediction.
    
    All features required for making a prediction.
    """
    ad_spend: float = Field(..., description="Advertising expenditure in thousands", ge=0)
    website_visits: float = Field(..., description="Number of website visitors", ge=0)
    average_product_price: float = Field(..., description="Average product price in dollars", ge=0)
    location_score: float = Field(..., description="Location quality score (0-100)", ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "ad_spend": 25.0,
                "website_visits": 5000.0,
                "average_product_price": 75.0,
                "location_score": 80.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for revenue prediction.
    
    Includes predicted revenue and feature contributions breakdown.
    """
    predicted_revenue: float = Field(..., description="Predicted monthly revenue in thousands")
    explanation: Dict[str, float] = Field(..., description="Feature contributions to the prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_revenue": 45.23,
                "explanation": {
                    "ad_spend_contribution": 20.0,
                    "website_visits_contribution": 10.0,
                    "average_product_price_contribution": 7.5,
                    "location_score_contribution": 40.0,
                    "bias_contribution": 10.0
                }
            }
        }


class TrainResponse(BaseModel):
    """
    Response schema for model training.
    
    Returns training results including final loss and learned parameters.
    """
    final_loss: float = Field(..., description="Final mean squared error after training")
    weights: list = Field(..., description="Learned weights (coefficients) for each feature")
    bias: float = Field(..., description="Learned bias (intercept)")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_loss": 25.1234,
                "weights": [0.78, 0.0019, 0.098, 0.52],
                "bias": 10.45,
                "message": "Model trained and saved successfully"
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information.
    
    Returns current model weights and bias without retraining.
    """
    weights: list = Field(..., description="Current model weights (coefficients)")
    bias: float = Field(..., description="Current model bias (intercept)")
    is_trained: bool = Field(..., description="Whether the model has been trained")
    
    class Config:
        json_schema_extra = {
            "example": {
                "weights": [0.78, 0.0019, 0.098, 0.52],
                "bias": 10.45,
                "is_trained": True
            }
        }


class SimulateResponse(BaseModel):
    """
    Response schema for What-If scenario simulation.
    
    Returns predictions for multiple scenarios in a single request.
    """
    scenarios: Dict[str, float] = Field(..., description="Dictionary of scenario names to predicted revenue values")
    
    class Config:
        json_schema_extra = {
            "example": {
                "scenarios": {
                    "standard": 45230.0,
                    "double_ad_spend": 67850.0,
                    "double_visits": 52340.0
                }
            }
        }

