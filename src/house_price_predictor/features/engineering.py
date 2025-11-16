# src/features/engineer.py
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import RobustScaler


class HousePricePreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,model_type:str='tree'):
        self.model_type = model_type
        self.scaler=None
        self.feature_names = None
    def fit(self,X:pd.DataFrame,y=None):
        """fit our data for preprocessing pipeline

        Parameters
        ----------
        X : pd.DataFrame
            data
        y : _type_, optional
            , by default None

        Returns
        -------
        """
        self.feature_names = X.columns.tolist()
        
        # check if model type is linear if so we need to fit a robustscaler for numeric features
        if self.model_type == 'linear':
            self.scaler = RobustScaler()
            numeric_cols = X.select_dtypes(include=[float,int]).columns
            for col in numeric_cols:
                X[col] = np.log1p(X[col])
            self.scaler.fit(X[numeric_cols])
        return self
    def transform(self,X:pd.DataFame,y=None) ->pd.DataFrame:
        """Transform or data using our preprocessing pipeline

        Parameters
        ----------
        X : pd.DataFame
            Data
        y : _type_, optional
            , by default None

        Returns
        -------
        pd.DataFrame
            transformed data
        """
        X = X.copy()
        # create additional features
        X = self._feature_engineering(X)
        if self.model_type == 'linear':
            X = self._transform_linear_type(X)
        else:
            X = self._transform_tree_type(X)
        X = self._convert_feats_to_binary(X)
        X = self._features_to_keep(X)
        return X
        
    def _transform_linear_type(self,X:pd.DataFrame) ->pd.DataFrame:
        """if model type is linear we will call this to scaler the
        numeric features by first using log transform as numeric features
        are skewed and then transform with a RobustScaler

        Parameters
        ----------
        X : pd.DataFrame
            data

        Returns
        -------
        pd.DataFrame
            transformed data
        """
        feats=['GrLivArea','TotalBsmtSF','total_sf']
        for feat in feats:
            X[feat]=np.log1p(X[feat])
        X[feats]=self.scaler.transform(X[feats])
        return X
    def _transform_tree_type(self,X:pd.DataFrame) ->pd.DataFrame:
        """if model type is not linear this is called but will just
        pass the data back

        Parameters
        ----------
        X : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return X
            
        
    def _feature_engineering(self,X:pd.DataFrame) ->pd.DataFrame:
        """perform feature engineering by creating additional featurees
        from the raw data

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        
        if 'YrSold' in self.feature_names:
            X['age_at_sale'] = X['YrSold'] - X['YearBuilt']
        else:
            # no YrSold found this implies data is inference data being passed and thus will
            # use todays year to indicate potential price if house sold this year
            today_date = dt.datetime.now()
            today_year = today_date.year
            X['age_at_sale'] = today_year = X['YearBuilt']
        X['age_of_house_squared'] = X['age_at_sale']**2
        if 'house_have_remodel' not in self.feature_names and 'YearRemodAdd' in self.feature_names:
            X['house_have_remodel']= np.where(X['YearBuilt']!=X['YearRemodAdd'],'yes','no')
        elif 'house_have_remodel' not in self.feature_names and 'YearRemodAdd' not in self.feature_names:
            X['house_have_remodel']='no'
        # ratio of finished basement sf to total basement sf
        X['ratio_finished_bsmt']=X['BsmtFinSF']/(X['TotalBsmtSF']+1)
        # create total full baths
        X['total_full_baths'] = X['FullBath'] + X['BsmtFullBath']
        # create total half baths
        X['total_half_baths'] = X['HalfBath'] + X['BsmtHalfBath']
        # ratio of of beds to bathrooms
        X['bed_bath_ratio'] = X['BedroomAbvGr']/X['total_full_baths']
        X['bed_bath_ratio'] = X['bed_bath_ratio'].replace([np.inf,-np.inf],np.nan).fillna(0)
        if 'GarageType' in self.feature_names:
            X['has_garage'] = np.where(~X['GarageType'].isnull(),'yes','no')
        if 'GarageFinish' in self.feature_names:
            X['garage_finished'] =np.where((X['GarageFinish']=='RFn')|
                                           (X['GarageFinish']=='Fin'),'yes',
                                           'no')
        # total square feet
        X['total_sf']= X['GrLivArea'] + X['TotalBsmtSF']
        # basement sf ratio to total scare feet
        X['basement_ratio'] = X['TotalBsmtSF'] / (X['total_sf']+1)
        # bedrooms per 1k square feet
        X['bedrooms_per_1ksf'] = X['BedroomAbvGr'] / (X['total_sf']/1000 + 1)
        
        return X
    def _features_to_keep(self,X:pd.DataFrame) ->pd.DataFrame:
        """Keep only the columns we need for modeling

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        
        cols_for_modeling = ['OverallQual','OverallCond','age_at_sale',
                             'age_of_house_squared','house_have_remodel',
                             'GrLivArea','TotalBsmtSF','total_sf',
                             'ratio_finished_bsmt','basement_ratio',
                             'total_full_baths','total_half_baths',
                             'BedroomAbvGr','bedrooms_per_1ksf',
                 'bed_bath_ratio','has_garage','garage_finsihed']
        X = pd.DataFrame(X,columns=cols_for_modeling)
        return X
    
    def _convert_feats_to_binary(self,X:pd.DataFrame) ->pd.DataFrame:
        """We have features with yes or no values and we want to 
        convert them to binary 1, 0 values

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        
        """
        
        X = X.copy()
        binary_feats = ['house_have_remodel','has_garage','garage_finished']
        for feat in binary_feats:
            X[feat]=np.where(X[feat].isin(['yes','Yes','Y',
                                           'y']),1,0)
        return X
    
            
    
        
    
            
                
        
        
        
        
        
        
        
        



