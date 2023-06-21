import joblib
import json 
import os 
from src.get_dataframe import read_params
import argparse
import numpy as np
import pandas as pd 
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

model_path = os.path.join("prediction_service","poduction_model.pkl") 

def edu_trans(df):
    # if df['education'] in ['basic.4y', 'basic.6y', 'basic.9y']:
    #     df['education'] = 'basic'
    df['education'] = df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic')
    return df 


def form_response(df):
    try:
        config = get_args_config()
        pre_processor_path = config['pipeline']['preprocessor_path']
        prod_model_path = config['ml_flow_config']['production_model_path']


        
        data = pd.DataFrame(df, columns=df.keys(), index=[0]).copy()
        data = edu_trans(data)
        # print(data)

        pre_model = joblib.load(pre_processor_path)
        prod_model = joblib.load(prod_model_path)

        pre_data = pre_model.transform(data)
        # print(pre_data)

        pred = prod_model.predict(pre_data)

        if pred.tolist()[0] == 0:
            val = "You're Not Eligible for the Loan"
        else:
            val = "Congrats...! You're Eligible for loan"

        return val

    except Exception as e:
        print(e)
        # error ={"error": "Something went wrong try again"}
        error = {"error": e}
        return error



def get_args_config():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)

    return config


