import argparse
import numpy as np
import os
import torch
import yaml

from src.utils.evaluate import evaluate, prediction, save_metrics, save_pred
from src.utils.logs import get_logger
from src.utils.train import COVID19Dataset, Model
from torch.utils.data import DataLoader

def main(config_path):
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger(log_level=config['base']['log_level'])
    file_path = 'data/processed'
    x_train = np.load(os.path.join(file_path, 'x_train.npy'))
    y_train = np.load(os.path.join(file_path, 'y_train.npy'))
    x_valid = np.load(os.path.join(file_path, 'x_valid.npy'))
    y_valid = np.load(os.path.join(file_path, 'y_valid.npy'))
    x_test = np.load(os.path.join(file_path, 'x_test.npy'))
    train_dataset, valid_dataset, test_dataset= COVID19Dataset(x_train, y_train), COVID19Dataset(x_valid, y_valid), COVID19Dataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=False, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['train']['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, pin_memory=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config['train']['save_path']))

    if not os.path.isdir('./report'):
        os.mkdir('./report')

    train_mse = evaluate(train_loader, model, device)
    valid_mse = evaluate(valid_loader, model, device)
    logger.info(f'train mse: {train_mse} / valid mse: {valid_mse}')
    save_metrics(train_mse, valid_mse, config['evaluate']['metrics_save_path'])
    preds = prediction(test_loader, model, device)
    save_pred(preds, config['evaluate']['pred_save_path'])
    
if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)