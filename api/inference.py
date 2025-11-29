import joblib
import pandas as pd
from datetime import datetime
from house_price_predictor.config.schemas import HousePredictionRequest, PredictionResponse
from house_price_predictor.config.house_price_core import MLFLOW_TRACKING_URI,LOG_DIR,config
import mlflow
import typing as t
import numpy as np

# load models
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"models:/{config.model_configs.model_name}/latest"
preprocessor_uri = f"models:/{config.model_configs.preprocessor_model_name}/latest"

try:
    model = mlflow.lightgbm.load_model(model_uri)
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    # Prepare input data
    input_data = pd.DataFrame([request.model_dump(mode='python')])
    
    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(processed_features)[0] # model returns an value in array shape
    
    # Convert numpy.float32 to Python float and round to 2 decimal places
    predicted_price = round(float(predicted_price), 2)
    
    feature_contri = model.predict(processed_features,pred_contrib=True)
    
    shape_featue_df = pd.DataFrame(feature_contri,
                                   columns = ['OverallQual','OverallCond','age_at_sale',
                                              'age_of_house_squared','house_have_remodel','GrLivArea',
                                              'TotalBsmtSF','total_sf','ratio_finished_bsmt','basement_ratio',
                                              'total_full_baths','total_half_baths','BedroomAbvGr',
                                              'bedrooms_per_1ksf','bed_bath_ratio','has_garage',
                                              'garage_finished','expected_value'])

    # Confidence interval (10% range)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    # Convert confidence interval values to Python float and round to 2 decimal places
    confidence_interval = [round(float(value), 2) for value in confidence_interval]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        shape_feature_contributions=shape_featue_df,
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[HousePredictionRequest]) -> pd.DataFrame:
    """
    Perform batch predictions.
    """
    input_data = pd.DataFrame()
    for req in requests:
        data = pd.DataFrame(req.model_dump(mode='python'))
        input_data = pd.concat([input_data,data],ignore_index=True,
                               sort=False)
    del data
    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make predictions
    predictions = model.predict(processed_features)
    # get shape feature contributions
    feature_contri = model.predict(processed_features,pred_contrib=True)
    
    prediction_df = pd.DataFrame(feature_contri,
                                   columns = ['OverallQual','OverallCond','age_at_sale',
                                              'age_of_house_squared','house_have_remodel','GrLivArea',
                                              'TotalBsmtSF','total_sf','ratio_finished_bsmt','basement_ratio',
                                              'total_full_baths','total_half_baths','BedroomAbvGr',
                                              'bedrooms_per_1ksf','bed_bath_ratio','has_garage',
                                              'garage_finished','expected_value'])
    prediction_df['predictions']=predictions
    
    return prediction_df