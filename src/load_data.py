import os 
from get_data_params import read_params, get_data 
import argparse 

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path
