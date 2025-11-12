# src/data/processor.py
import pandas as pd
import numpy as np
import logging
import typing as t




def clean_data(df:pd.DataFrame) ->pd.DataFrame:
  
    
    
    # Make a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    cols = cleaned_df.columns.tolist()
    
    for col in cols:
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count >0:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val,inplace=True)
            else:
                # means its of string type fill will mode
                mode_val = cleaned_df[col].mode()
                cleaned_df[col].fillna(mode_val,inplace=True)
    return cleaned_df
def fix_data_types_to_int(df:pd.DataFrame,cols:t.List[str]) ->pd.DataFrame:
    
    cleaned_df = df.copy()
    for col in cols:
        cleaned_df[col]=cleaned_df[col].astype(int)
    return cleaned_df    
    
