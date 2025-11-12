import pandas as pd
import numpy as np
import sys,os
from house_price_predictor.config.house_price_core import ROOT
def process_test():
    print(f'Workding directory is: {os.getcwd()}')
    if 'PYTHONPATH' in os.environ:
        print('the python path key was found')
    else:
        print('not found')
    print(f'python path is: {os.getenv("PYTHONPATH")}')
    print(sys.path)
    print(f'The projecty root is: {ROOT}')
    
if __name__=="__main__":
    process_test()
    
