import pandas as pd
import numpy as np
import sys, os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,LOG_DIR,MLFLOW_TRACKING_URI,config
from house_price_predictor.data_management.process_raw_data import clean_data,fix_data_types_to_int
from house_price_predictor.utils.house_price_logging import HousePriceLogger, house_price_logger,default_logger


@house_price_logger(HousePriceLogger(log_file=f'{LOG_DIR}/{config.app_configs.house_price_log_file}'))
def clean_data_pipeline():
    print('Begingging the ETL process of clean the raw data')
    raw_data = pd.read_csv(DATA_DIR /config.app_configs.raw_data_file)
    clean_df = clean_data(raw_data,config.model_configs.columns_to_rename)
    clean_df = fix_data_types_to_int(df=clean_df,cols=config.model_configs.columns_to_convert_to_int)
    print('saving Processed data')
    clean_df.to_pickle(DATA_DIR/config.app_configs.clean_data_file)
    raw_shape = raw_data.shape
    process_shape = clean_df.shape
    return {'raw_shaped':raw_shape,
            'processed_data_shape':process_shape}
    
if __name__=="__main__":
    clean_etl_results = clean_data_pipeline()
    