import joblib
import pandas as pd 
import numpy as np
import os 
import sys 

# from src import get_dataframe as gdf
from get_dataframe import read_params, get_data

from sklearn import set_config
set_config(transform_output="pandas")
# READ THE DATA FROM DATA SOURCE
# SAVE IT IN THE DATA/RAW FOR FURTHER PROCESS
# from src import get_data_params
# from . import get_data_params.read_params, get_data_params.get_data
import argparse

def edu_trans(df):
    df['education'] = df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic')
    return df 

def preprocessing(config_path):

    # cd = CustomException()
    config = read_params(config_path)
    df = get_data(config_path)
    processed_data_path = config["data_source"]["preprocessed_data_source"]

    preprocessor_path = config['pipeline']['preprocessor_path']

    processor = joblib.load(preprocessor_path)

    processed_data = processor.fit_transform(df)

    processed_data.to_csv(processed_data_path, sep=",", index=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocessing(config_path=parsed_args.config)


