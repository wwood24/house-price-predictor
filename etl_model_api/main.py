import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,config
from house_price_predictor.data_management.process_raw_data import clean_data,fix_data_types_to_int
from house_price_predictor.features.engineering import HousePricePreprocessor
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from house_price_predictor.build_model.train_model import HousePriceProductionModel

def etl_model_training():
    print('Beginning the ETL Process')
    raw_data = pd.read_csv(DATA_DIR /config.app_configs.raw_data_file)
    print(f'raw data shape is: {raw_data.shape}')
    print('Cleaning Data')
    clean_df = clean_data(raw_data,config.model_configs.columns_to_rename)
    clean_df = fix_data_types_to_int(df=clean_df,cols=config.model_configs.columns_to_convert_to_int)
    print('saving Processed data')
    clean_df.to_pickle(DATA_DIR/config.app_configs.clean_data_file)
    
def preprocessing_training_pipeline():
    
    clean_df = pd.read_pickle(DATA_DIR/config.app_configs.clean_data_file)
    
    # split data into training and testing
    y = clean_df[config.model_configs.target_variable]
    X = clean_df.drop(config.model_configs.target_variable,axis=1)
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                     test_size=0.2,
                                                     random_state=42)
    
    preprocessor = HousePricePreprocessor(model_type='tree')
    
    # setup Mflow tracking
    mlflow.set_tracking_uri(config.app_configs.mlflow_tracking_uri)
    exp_id = config.app_configs.mlflow_experiment_id
    
    with mlflow.start_run(experiment_id=exp_id,
                          run_name='preprocessor_pipeline') as run:
        
    
        preprocessor.fit(x_train)
        mlflow.sklearn.log_model(sk_model=preprocessor,
                                 artifact_path=config.model_configs.preprocess_pipeline_name)
        x_train_transformed = preprocessor.transform(x_train)
        x_test_transformed = preprocessor.transform(x_test)
    print(f'the run info is: {run.info.artifact_uri}')
    print(f'the run other info is {run.info.run_id}')
    # save data
    x_train_transformed.to_pickle(DATA_DIR/config.app_configs.train_file)
    x_test_transformed.to_pickle(DATA_DIR/config.app_configs.test_file)
    y_train.to_pickle(DATA_DIR/config.app_configs.train_y_file)
    y_test.to_pickle(DATA_DIR/config.app_configs.test_y_file)
    
def train_model():
    # load data
    x_train=pd.read_pickle(DATA_DIR/config.app_configs.train_file)
    x_test=pd.read_pickle(DATA_DIR/config.app_configs.test_file)
    y_train=pd.read_pickle(DATA_DIR/config.app_configs.train_y_file)
    y_test=pd.read_pickle(DATA_DIR/config.app_configs.test_y_file)
    
    # set up model
    
    house_price_model = HousePriceProductionModel(mlflow_track_uri=config.app_configs.mlflow_tracking_uri,
                                                  exp_id=config.app_configs.mlflow_experiment_id,
                                                  run_name=config.app_configs.mlflow_run_name)
    
    model = house_price_model.get_model_instance(model_name=config.model_configs.best_model,
                                                 params=config.model_configs.best_model_params)
    print('Train and log model')
    house_price_model.train_log_model_experiment(x_train=x_train,x_test=x_test,
                                                 y_train=y_train,y_test=y_test)
    print('completed training and loging model')
    
        
  
    
if __name__=="__main__":
    # etl_model_training()
    # preprocessing_training_pipeline()
    train_model()
    
