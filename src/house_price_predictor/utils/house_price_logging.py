import logging
import typing as t
from functools import wraps
from pathlib import Path
import datetime as dt

class HousePriceLogger:
    def __init__(self,log_file:t.Union[str,Path]):
        self.log_file = log_file
    
    def get_logger(self,logger_name:str,log_file:t.Union[str,Path]) -> logging.Logger:
        logger = logging.getLogger(name=logger_name)
        logger.setLevel(level=logging.INFO)
        
        f_handler = logging.FileHandler(filename=self.log_file)
        formatter = logging.Formatter(fmt=f'%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(message)s')
        f_handler.setFormatter(formatter)
        
        logger.addHandler(f_handler)
        return logger
    
    

def default_logger() ->logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    return logger
    
    
    
    
def house_price_logger(logger:t.Union[HousePriceLogger,logging.Logger]):
    def setup_logger(func):
        if isinstance(logger,HousePriceLogger):
            logger = logger.get_logger(logger_name=f'{func.__name__}')
        else:
            logger = default_logger()
        @wraps(func)
        def wrapper(*args,**kwargs):
            start_time = dt.datetime.now()
            logger.info(msg=f'Starting {func.__name__} method')
            try:
                results = func(*args,**kwargs)
                end_time = dt.datetime.now()
                run_time = end_time-start_time
                logger.info(f'Completed {func.__name__} in {run_time}')
                return results
            except Exception as e:
                end_time = dt.datetime.now()
                run_time = end_time-start_time
                logger.error(f'{func.__name__} failed to complete in {run_time} with {e} as error')
        return wrapper
    return setup_logger
            
            
            
        
    