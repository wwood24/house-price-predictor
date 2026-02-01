import joblib
import pandas as pd
from datetime import datetime
from house_price_predictor.config.schemas import HousePredictionRequest, PredictionResponse
from house_price_predictor.config.house_price_core import MLFLOW_TRACKING_URI,LOG_DIR,config
import mlflow
import typing as t
import numpy as np
from house_price_predictor.utils.house_price_logging import HousePriceLogger,house_price_logger

# load models
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"models:/{config.model_configs.model_name}/latest"
preprocessor_uri = f"models:/{config.model_configs.preprocessor_model_name}/latest"

try:
    model = mlflow.lightgbm.load_model(model_uri)
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

@house_price_logger(HousePriceLogger(log_file=f'{LOG_DIR}/inference.log'))
def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    try:
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
                                              'garage_finished','expected_value'],index=[0])
    
    
        feature_contri_dict = shape_featue_df.to_dict(orient='records')[0]

        # Confidence interval (10% range)
        confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

        # Convert confidence interval values to Python float and round to 2 decimal places
        confidence_interval = [round(float(value), 2) for value in confidence_interval]

        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            feature_contribution=feature_contri_dict,
            prediction_time=f'{datetime.now().isoformat()}',
            status='success'
        
        )
    except Exception as e:
        return PredictionResponse(
            predicted_price=0,
            confidence_interval=0,
            feature_contribution={"error":0},
            prediction_time=f'{datetime.now().isoformat()}',
            status=f'{str(e)}'
        )
@house_price_logger(HousePriceLogger(log_file=f'{LOG_DIR}/inference.log'))
def batch_predict(requests: list[HousePredictionRequest]) -> t.List[PredictionResponse]:
    """
    Perform batch predictions.
    """
    results = []
    for req in requests:
        data = pd.DataFrame([req.model_dump(mode='python')])
        try:
            # Preprocess input data
            processed_features = preprocessor.transform(data)
            # Make predictions
            predicted_price = model.predict(processed_features)[0]
            # round to numpy with two decimals
            predicted_price = round(float(predicted_price),2)
            # get shape feature contributions
            feature_contri = model.predict(processed_features,pred_contrib=True)
            feature_contri_dict = feature_contri.to_dict(orient='records')[0]
            # Confidence interval (10% range)
            confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]
            # Convert confidence interval values to Python float and round to 2 decimal places
            confidence_interval = [round(float(value), 2) for value in confidence_interval]
            results.append(
                PredictionResponse(
                    predicted_price=predicted_price,
                    confidence_interval=confidence_interval,
                    feature_contribution=feature_contri_dict,
                    prediction_time=f'{datetime.now().isoformat()}',
                    status='success'
                )
            )
        except Exception as e:
            results.append(
                PredictionResponse(
                    predicted_price=0,
                    confidence_interval=[0,0],
                    feature_contribution={'error':0},
                    prediction_time=f'{datetime.now().isoformat()}',
                    status=f'{str(e)}'
                )
            )
            
    
 
    
    return results