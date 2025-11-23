import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from mlflow.tracking import MlflowClient
import sklearn
import lightgbm as lgb
import typing as t
import mlflow.xgboost
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mlflow.models import infer_signature


class HousePriceProductionModel:
    def __init__(self,mlflow_track_uri:str,exp_id:int,run_name:str,model_prod_name):
        self.mlflow_tracking_uri = mlflow_track_uri
        self.experiment_id = exp_id
        self.run_name = run_name
        self.model = None
        self.model_params = None
        self.prod_model_name = model_prod_name
        
    def get_model_instance(self,model_name:str,params:t.Dict[str,t.Any]):
        model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor,
        'LightGBM':lgb.LGBMRegressor
        }
        self.model_params = params
        if 'learning_rate' in self.model_params.keys():
            self.model_params['learning_rate']=float(params.get('learning_rate'))
        if 'n_estimators' in self.model_params.keys():
            self.model_params['n_estimators']=int(params.get('n_estimators'))
        if 'num_leaves' in self.model_params.keys():
            self.model_params['num_leaves'] = int(params.get('num_leaves'))
        if 'n_jobs' in self.model_params.keys():
            self.model_params['n_jobs'] = int(params.get('n_jobs'))
        if model_name not in model_map.keys():
            raise ValueError('Invalid model name provided')
        self.model = model_map[model_name](**self.model_params)
        return self.model
    
    def train_log_model_experiment(self,x_train:pd.DataFrame,x_test:pd.DataFrame,
                                   y_train:pd.Series,y_test:pd.Series,model_prod_name:str):
        
        # set up mlflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        with mlflow.start_run(experiment_id=self.experiment_id,
                              run_name=self.run_name) as model_run:
            self.model.fit(x_train,y_train)
            
            signature = infer_signature(x_train,self.model.predict(x_train))
            
            # log metrics from predicts
            y_pred = self.model.predict(x_test)
            
            mae = mean_absolute_error(y_test,y_pred)
            rmse = root_mean_squared_error(y_test,y_pred)
            r2 = r2_score(y_test,y_pred)
            
            metrics = {'mae':mae,
                       'rmse':rmse,
                       'r2_score':r2}
            mlflow.log_metrics(metrics=metrics)
            mlflow.log_params(self.model_params)
            
            if isinstance(self.model,lgb.LGBMRegressor):
                mlflow.lightgbm.log_model(self.model,"tuned_model",signature=signature,
                                          registered_model_name=self.prod_model_name)
                feature_importance = self.model.feature_importances_
                feature_names = self.model.feature_name_
            elif isinstance(self.model,xgb.XGBRegressor):
                mlflow.xgboost.log_model(self.model,"tuned_model",signature=signature,
                                         registered_model_name=self.prod_model_name)
            else:
                mlflow.sklearn.log_model(self.model,'tuned_model',signature=signature,
                                         registered_model_name=self.prod_model_name)
                
                
            # make some plots
            residuals = y_test - y_pred
            
            fig,ax = plt.subplots(figsize=(12,10))
            sns.scatterplot(x=y_pred,y=residuals,ax=ax)
            ax.set_xlabel('predicated value')
            ax.set_ylabel('Residuals')
            ax.set_title('Pred vs Residuals')
            mlflow.log_figure(fig,"pred_vs_residuals.png")
            plt.close(fig)
            
            
            ## qq plot
            fig = sm.qqplot(residuals,line='s')
            plt.title('QQ Plot of Residiuals')
            mlflow.log_figure(fig,'qq_residuals.png')
            plt.close(fig)
            
            # feature importances
            fig,ax=plt.subplots(figsize=(12,10),ncols=1,nrows=1)
            sns.barplot(x=feature_names,y=feature_importance,ax=ax)
            plt.xticks(rotation=45)
            ax.set_xlabel('features')
            ax.set_ylabel('feature imporance')
            mlflow.log_figure(fig,'feature_importance.png')
            plt.close(fig)
            
            fig,ax=plt.subplots(figsize=(12,10),ncols=1,nrows=1)
            sns.scatterplot(x=y_test,y=y_pred,ax=ax)
            ax.set_xlabel('acutal values')
            ax.set_ylabel('predicted values')
            ax.set_title('Acutual values vs predicted')
            mlflow.log_figure(fig,'actual_vs_predicted.png')
            plt.close(fig)
            
        return model_run
    
    def move_model_to_production(self,model_version:int,model_stage:str):
        client = MlflowClient()
        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=self.prod_model_name,
        version=model_version,stage=model_stage,archive_existing_versions=True)
        features = ['OverallQual','OverallCond','age_at_sale',
            'age_of_house_squared','house_have_remodel','GrLivArea','TotalBsmtSF',
            'total_sf','ratio_finished_bsmt','basement_ratio','total_full_baths',
            'total_half_baths','BedroomAbvGr','bedrooms_per_1ksf','bed_bath_ratio',
            'has_garage,garage_finished']
        # Add a human-readable description
        description = (
            f"Model for predicting house prices.\n"
             f"Features used: {features} \n"
            f"Target variable: SalePrice\n"
             )
        client.update_registered_model(name=self.prod_model_name, description=description)

        # Add dependency tags
        deps = {
             "scikit_learn_version": sklearn.__version__,
             "lightbm_version":lgb.__version__,
             "xgboost_version": xgb.__version__,
             "pandas_version": pd.__version__,
            "numpy_version": np.__version__}
        for k, v in deps.items():
            client.set_registered_model_tag(self.prod_model_name, k, v)
        
        return 


