from pathlib import Path
from pydantic import BaseModel
from strictyaml import load, YAML
import typing as t
import house_price_predictor
from dotenv import load_dotenv
import os

load_dotenv()



PROJECT_ROOT = Path(house_price_predictor.__file__).resolve().parent
ROOT = PROJECT_ROOT.parent
DATA_DIR = os.getenv('DATA_DIR','/app/data')
CONFIG_FILE_PATH = PROJECT_ROOT /'house_price_config.yml'

class AppConfig(BaseModel):
    package_name: str
    mlflow_tracking_uri: str
    mlflow_experiment_id: int
    raw_data_file: str
    clean_data_file: str
    
class ModelConfig(BaseModel):
    columns_to_convert_to_int:t.List[str]
    target_variable: str
    features_set:t.List[str]
    binary_features:t.List[str]
    linear_scaled_features:t.List[str]
    preprocess_pipeline_name: str
    model_name: str
    best_model:str
    best_model_params:t.Dict[str,t.Any]
    
    
class Config(BaseModel):
    app_config = AppConfig
    model_config = ModelConfig
    

def find_config_file()->Path:
    """Find our config file within our project.

    Returns
    -------
    Path
        Full location of our config file

    Raises
    ------
    Exception
        Provides error if the file does not exist
    """
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f'Config file not found at {CONFIG_FILE_PATH}')

def fetch_configs_from_yaml(cfg_path:Path=None)->YAML:
    """Fetches the configs from a yaml file and returns an YAML object.

    Parameters
    ----------
    cfg_path : Path, optional
        location of config file, by default None

    Returns
    -------
    YAML
        parsed data from yaml file

    Raises
    ------
    OSError
        raises error if can't read file
    """
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path,'rb') as file:
            parse_configs = load(file.read())
            return parse_configs
    raise OSError(f'unable to parse configs from {cfg_path}')

def create_and_validate_configs(parsed_configs:YAML=None) ->Config:
    """Parses our data from yaml info our Config object.

    Parameters
    ----------
    parsed_configs : YAML, optional
        _description_, by default None

    Returns
    -------
    Config
        object build on pydandic basemodel with two keys app_config and model_config.
    """
    
    if parsed_configs is None:
        parsed_configs = fetch_configs_from_yaml()
        
    _config= Config(
        app_config = AppConfig(**parsed_configs.data),
        model_config = ModelConfig(**parsed_configs.data)
    )
    
    return _config

config = create_and_validate_configs()

    
        
    
    

