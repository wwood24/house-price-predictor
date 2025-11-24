# House Price Predictor Package

This package will contain the code required to process train and log models into mlflow.

## package Structure

house_price_predictor has the followiong subpackages

- build_model:
  - train_model.py: this has the class object that will use the config to build and train the final model and log it into mlflow
- config
  - house_price_core.py: using pydantic to build a config class object that has the key values of the package as key value pairs
  - schemas.py: used to define the schemas required for inference and output 
- data_management:
 - process_raw_data.py: process the raw data
- features:
 - engineering.py: has the code to build the preprocessor pipeline.
- utils:
 - house_price_logging.py: a logging decorator to log the the function calls
- house_price_config.yml: has the apps configs that is used in the house_price_core.py


all these functions and classes will be use din the etl_model_api and the the models produced and stored in mlflow will be used in the inference api