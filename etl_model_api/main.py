import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,config
from house_price_predictor.data_management.process_raw_data import clean_data,fix_data_types_to_int
def etl_model_training():
    print('Beginning the ETL Process')
    raw_data = pd.read_csv(DATA_DIR /config.app_configs.raw_data_file)
    print(f'raw data shape is: {raw_data.shape}')
    print('Cleaning Data')
    clean_df = clean_data(raw_data)
    clean_df = fix_data_types_to_int(df=clean_df,cols=config.model_configs.columns_to_convert_to_int)
    print('saving Processed data')
    clean_df.to_pickle(DATA_DIR/config.app_configs.clean_data_file)
        
  
    
if __name__=="__main__":
    etl_model_training()
    
