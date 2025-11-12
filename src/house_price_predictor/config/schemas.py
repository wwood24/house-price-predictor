from pydantic import BaseModel, Field
import typing as t

class HousePredictionRequest(BaseModel):
    YrSold: t.Optional[int] = Field(...,default=None,description='Year House Was Sold')
    YearBuilt: int = Field(...,gt=1800,description='Year House was built')
    OverallQual: int = Field(...,ge=1,le=10,description='Overall Quality of House material as a scale of 1-10')
    OverallCond: int = Field(...,ge=1,le=10,description='Overall Condition of House as scale of 1-10')
    age_at_sale: int = Field(...,ge=0,description='Age of House at time of Sale')
    age_of_house_squared: int = Field(...,ge=0,description='Squared value of age_at_sale')
    house_have_remodel: int = Field(...,ge=0,le=1,description='Binary value indicating if the house had any remodels done.')
    GrLivArea: float = Field(..., gt=0, description="Square footage of the house above Ground")
    TotalBsmtSF: float = Field(...,ge=0,description='Total Square feet of Basement')
    total_sf: float = Field (...,gt=0,description='Total Square feet of place. Sum of GrLivArea and TotalBsmtSF')
    ratio_finished_bsmt: float = Field(...,ge=0,le=1,description='Percentage of Basement that is finished')
    basement_ratio: float = Field(...,ge=0,le=1,description='Percentage of Basement Square Feet to total Square Feet')
    total_full_baths: int = Field(..., ge=0, description="Number of Full Baths")
    total_half_baths: int = Field(...,ge=0,description='Number of Half Baths')
    BedroomAbvGr: float = Field(..., ge=0, description="Number of bathrooms")
    bed_bath_ratio: float = Field(...,ge=0,le=1,description='Ratio of bedrooms to bathrooms')
    bedrooms_per_1ksf: float = Field(...,ge=0,description='Number of Bedrooms per 1000 square feet')
    has_garage: int = Field(...,ge=0,le=1,description='Does place have a garage or not')
    garage_finished: int = Field(...,ge=0,le=1,description='Is the Garage finished or not')
    
    

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: t.List[float]
    features_importance: t.Dict[str,float]
    prediction_time: str