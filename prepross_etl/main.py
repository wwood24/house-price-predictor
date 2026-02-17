import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,LOG_DIR,MLFLOW_TRACKING_URI,config
from house_price_predictor.data_management.process_raw_data import clean_data,fix_data_types_to_int
from house_price_predictor.features.engineering import HousePricePreprocessor
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from house_price_predictor.utils.house_price_logging import HousePriceLogger, house_price_logger,default_logger


@house_price_logger(HousePriceLogger(log_file=f'{LOG_DIR}/{config.app_configs.house_price_log_file}'))
def run_preprocessing_pipeline():
    clean_df = pd.read_pickle(DATA_DIR/config.app_configs.clean_data_file)
    
    # split data into training and testing
    y = clean_df[config.model_configs.target_variable]
    X = clean_df.drop(config.model_configs.target_variable,axis=1)
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                     test_size=0.2,
                                                     random_state=42)
    
    preprocessor = HousePricePreprocessor(model_type='tree')
    
    # setup Mflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp_id = config.app_configs.mlflow_experiment_id
    
    
    with mlflow.start_run(experiment_id=exp_id,
                          run_name='preprocessor_pipeline') as run:
        
        
        preprocessor.fit(x_train)
        mlflow.sklearn.log_model(sk_model=preprocessor,input_example=x_train,
                                 artifact_path=config.model_configs.preprocess_pipeline_name,
                                 registered_model_name=config.model_configs.preprocessor_model_name)
        x_train_transformed = preprocessor.transform(x_train)
        x_test_transformed = preprocessor.transform(x_test)
    # set up client info for model
    
    # save data
    x_train_transformed.to_pickle(DATA_DIR/config.app_configs.train_file)
    x_test_transformed.to_pickle(DATA_DIR/config.app_configs.test_file)
    y_train.to_pickle(DATA_DIR/config.app_configs.train_y_file)
    y_test.to_pickle(DATA_DIR/config.app_configs.test_y_file)
    # transition model
    client = MlflowClient()
    
    registered_model = client.get_registered_model(name=config.model_configs.preprocessor_model_name)
    model_version = registered_model.latest_versions[0] # get latest
    
    client.set_registered_model_alias(
        name=registered_model.name,
        version=model_version.version,
        alias=config.model_configs.preprocessor_alias
    )
    description = 'This is an sklearn preprocessing pipeline that performs the cleaning, feature engineering and data transforms.'
    client.update_registered_model(
        name=config.model_configs.preprocessor_model_name,
        description=description
    )
    # call get model info to retrieve the current statge and model version for logs
    registered_model = client.get_registered_model(name=config.model_configs.preprocessor_model_name)
    model_version = registered_model.latest_versions[0] # get latest
    print(f'To view the mlflow run for this is is the key info: uri:{run.info._artifact_uri},run_id:{run.info.run_id},model_name:{registered_model.name},version:{model_version}')
    return {'preprocessor_uri': run.info.artifact_uri,
            'preprocessor_run_id':run.info.run_id,
            'model_name':registered_model.name,
            'model_version':model_version,
            'model_alias':config.model_configs.preprocessor_alias}
    