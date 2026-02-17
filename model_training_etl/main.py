import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,LOG_DIR,MLFLOW_TRACKING_URI,config
import mlflow
from mlflow.tracking import MlflowClient
from house_price_predictor.build_model.train_model import HousePriceProductionModel
from house_price_predictor.utils.house_price_logging import HousePriceLogger, house_price_logger,default_logger


@house_price_logger(HousePriceLogger(log_file=f'{LOG_DIR}/{config.app_configs.house_price_log_file}'))
def train_model_pipeline():
    # load data
    x_train=pd.read_pickle(DATA_DIR/config.app_configs.train_file)
    x_test=pd.read_pickle(DATA_DIR/config.app_configs.test_file)
    y_train=pd.read_pickle(DATA_DIR/config.app_configs.train_y_file)
    y_test=pd.read_pickle(DATA_DIR/config.app_configs.test_y_file)
    
    # set up model
    
    house_price_model = HousePriceProductionModel(mlflow_track_uri=MLFLOW_TRACKING_URI,
                                                  exp_id=config.app_configs.mlflow_experiment_id,
                                                  run_name=config.app_configs.mlflow_run_name,
                                                  model_prod_name=config.model_configs.model_name)
    
    model = house_price_model.get_model_instance(model_name=config.model_configs.best_model,
                                                 params=config.model_configs.best_model_params)
    print('Train and log model')
    active_run = house_price_model.train_log_model_experiment(x_train=x_train,x_test=x_test,
                                                 y_train=y_train,y_test=y_test)
    model_info = house_price_model.move_model_to_production(model_alias=config.model_configs.ml_model_alias)
    print('completed training and loging model')
    
    return {'model_uri':active_run.info.artifact_uri,
            'model_run_id':active_run.info.run_id} | model_info