import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT,DATA_DIR,config
def process_test():
    print(f'Workding directory is: {os.getcwd()}')
    print(f'python path is: {os.getenv("PYTHONPATH")}')
    print(sys.path)
    print(f'The projecty root is: {ROOT}')
    print(f'data dir is: {DATA_DIR}')
    print(f'model name is: {config.model_configs.model_name}')
    
if __name__=="__main__":
    process_test()
    
