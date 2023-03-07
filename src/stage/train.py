import argparse
import numpy as np
import os
import torch
import yaml

from src.utils.logs import get_logger
from src.utils.train import COVID19Dataset, Model, model_training, plot_loss_history, same_seed
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
    train_dataset, valid_dataset = COVID19Dataset(x_train, y_train), COVID19Dataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['train']['batch_size'], shuffle=True, pin_memory=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    same_seed(config['base']['random_state'])
    model = Model(input_dim=x_train.shape[1], unit=config['train']['unit']).to(device)
    loss_history = model_training(train_loader, valid_loader, model, config, device, logger)
    plot_loss_history(loss_history, config)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)