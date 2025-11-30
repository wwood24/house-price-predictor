from pydantic import BaseModel, Field
import typing as t
import pandas as pd

class HousePredictionRequest(BaseModel):
    YearBuilt: int = Field(...,gt=1800,description='Year House was built')
    OverallQual: int = Field(...,ge=1,le=10,description='Overall Quality of House material as a scale of 1-10')
    OverallCond: int = Field(...,ge=1,le=10,description='Overall Condition of House as scale of 1-10')
    house_have_remodel: t.Literal['yes','no'] = Field(...,description='Has the house have any remodels done')
    BsmtFinSF: float = Field(...,ge=0, description='Square footage of finished basement.')
    TotalBsmtSF: float = Field(...,ge=0,description='Total Square feet of Basement')
    GrLivArea: float = Field (...,gt=0,description='Total Square feet of living place above ground.')
    FullBath: int = Field(...,ge=0,description='Number of Full Baths above Ground.')
    BsmtFullBath: int = Field(...,ge=0,description='Number of Full Baths in basement')
    HalfBath: int = Field(...,ge=0,desciption='Number of Half baths above ground.')
    BsmtHalfBath: int = Field(...,ge=0,description='Number of Half Baths in basement')
    BedroomAbvGr: float = Field(..., ge=0, description="Number of bathrooms")
    has_garage: t.Literal['yes','no'] = Field(...,description='Does place have a garage or not')
    garage_finished: t.Literal['yes','no'] = Field(...,description='Is the Garage finished or not')
    
    

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: t.List[float]
    features_importance: t.Dict[str,float]
    prediction_time: str