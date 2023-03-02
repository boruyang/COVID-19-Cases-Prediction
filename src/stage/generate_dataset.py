import argparse
import numpy as np
import os
import pandas as pd
import yaml

from src.utils.generate_dataset import select_feature, train_valid_split
from src.utils.logs import get_logger

def main(config_path):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger(log_level=config['base']['log_level'])
    train_data = pd.read_csv('data/raw/covid_train.csv').values
    test_data = pd.read_csv('data/raw/covid_test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['generate_dataset']['valid_ratio'], config['base']['random_state'])
    logger.info(f'train_data size: {train_data.shape}, valid_data size: {valid_data.shape}, test_data size: {test_data.shape}')
    
    x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, config['generate_dataset']['feature_selection'])
    logger.info(f'number of features: {x_train.shape[1]}')

    for file_name, array in zip(['x_train', 'x_valid', 'x_test', 'y_train', 'y_valid'], [x_train, x_valid, x_test, y_train, y_valid]):
        os.system('mkdir -p data/processed')
        np.save(os.path.join('data', 'processed', file_name), array)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)