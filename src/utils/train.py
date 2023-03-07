import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from tqdm import tqdm

class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

def same_seed(seed: int): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Model(nn.Module):
    def __init__(self, input_dim, unit):
        super(Model, self).__init__()
        layers = [nn.Linear(input_dim, unit[0]),
                  nn.ReLU()]
        for i in range(1, len(unit)):
            layers.append(nn.Linear(unit[i - 1], unit[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(unit[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

def model_training(train_loader, valid_loader, model, config, device, logger):
    '''Model training process'''

    criterion = nn.MSELoss(reduction='mean')

    lr = config['train']['learning_rate']
    l2_penalty = config['train']['l2_penalty']
    if config['train']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_penalty, momentum=0.7)
    elif config['train']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    elif config['train']['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_penalty, momentum=0.7)

    if not os.path.isdir('./model'):
        os.mkdir('./model')
    
    n_epochs, best_loss, early_stop_count = config['train']['n_epochs'], math.inf, 0
    loss_record_epoch = []

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1} / {n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()
        loss_record = []

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
        
        mean_valid_loss = sum(loss_record) / len(loss_record)
        logger.info(f'Epoch [{epoch + 1} / {n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        loss_record_epoch.append((mean_train_loss, mean_valid_loss))

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['train']['save_path'])
            logger.info('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['train']['early_stop']:
            logger.info('\nModel is not improving, so we halt the training session.')
            return loss_record_epoch       
    
    return loss_record_epoch

def plot_loss_history(loss_history, config):
    loss_history = list(zip(*loss_history))
    plt.plot(range(config['train']['n_epochs']), loss_history[0], label='train loss')
    plt.plot(range(config['train']['n_epochs']), loss_history[1], label='validation loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(config['train']['history_save_path'])

