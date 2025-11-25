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

## Mlflow Setup

For this project its using a local version on the system as it set up with a backend postgress database to store metada for all projects other then this project.

- within local builds the MLFLOW_TRACKING_URI for any project will set to local host and port 5000
- since this code will be put into a podman container and needs to be able to communicate with this version of mlflow need set the MLFLOW_TRACKING_URI as en enviornment variable when running podman and have it set to **http://host.containers.internal:5000**

### Key commands of mlflow

- **mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)**:
 - here we want to set it to our local running instance of mlflow if we are not using containers or pass in the above value if we are having this code ran within a container. This is used to set the location of the mlflow store to store and retrieve experiments and metadata of our expiement.
 - **with mlflow.start_run() as run**:
  - this will start a new active run and assign this run to variable **run**
  - it can take the following key paramaters:
   - **run_id**: str, optional specificies unique id for the run. Can be useful to resume a previous run
   - **experiment_id**: str, optional.  Associates the run with a specific mlflow experiment.  Thus if one wants this run to be part of a previous set of experiments providing this value assures that this run is stored within that given experiment along with other experiments of same project. This is a good practice if one is testing many models for a given project thus grouping them within same experiment will make it easier to find and select best model to promote.
   - **run_name**: str,optional.  Assigns a huma readable name for the run.  Thus within the experiment id within the ui this is the name that will show up for this given experiment.
- **mlfow.log_metrics**:
 - this takes in a dictionary of key value pairs where the key is a specific metric and the value is the value of htis metric that we want to log for the experiment.  For classification this could be precision, recall, f1, accuracy.  For regression could be mae, rmse, r2_score, etc.  If one wants to do one at a time could also use **mlflow.log_metric**.
- **mlflow.log_params**:
  - this takes in a dictionary of key value paris of the models params one wants to log.  This is good to compare models params used.  Could be n_estimators for tree based models, etcs.
- **mlflow.model_flavor.log_model**:
 - call this will log the model within mlflow.  model_flavor is the the type of model being built could be sklearn, xgboost, lightgbm, etc. Using this will allow mlflow to better save the model correctly i.e. building a xgboost model should use the xgboost flavor and not sklearn. Key paramters to fill in when calling this:
  - **model**: this is the model wanted to be logged.
  - **artifact_path**: str, the run relative path where the model will be saved. 
  - **signature**: an mlflow.models.ModelSignature object that defines the models inputs and outputs.  This helps with model deployment and validation.
  - **registered_model_name**: str, optional.  If provided the model will also be registered within the Model Registry under this name.

- **mlflow.log_figure()**:
 - this will allow one to log plots and figures into and artifact path to be displayed in the ui.  Could be precision vs recall. Confusion matrix, etc.
## Using MlflowClient

Would use this to get info of an regsitred model in the model registry and or update the registered model with other info not provided whe setting the model to the model registry when using **registered_model_name** when using mlflow.log_model from above.

some key features/mthods one can call with client is

- **get_registered_model()**: use this if want to get model my name as the only paramter need to supply is the name of model. With this one can do the following:
```python
client = MlflowClient()
registered_model = client.get_registered_model(model_name)
print(f'{registered_model.name})
print(f'{registered_model.version})
```






