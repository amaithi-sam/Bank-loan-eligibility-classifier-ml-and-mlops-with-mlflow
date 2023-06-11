import os 
import yaml 
import pandas as pd 
import argparse 


def read_params(config_path):
    '''
    Reads the params.yaml file  and return a dictionary with parameters and it's values
    '''
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config 

def get_data(config_path):
    '''
    read the CSV file from the local directory and return a pandas dataframe
    '''
    config = read_params(config_path)
    
    data_path = config['data_source']['local_data_source']

    df = pd.read_csv(data_path, sep=',')

    return df.sample(n=2500)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    # data = get_data(parsed_args.config)