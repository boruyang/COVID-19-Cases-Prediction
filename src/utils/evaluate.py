import csv
import json
import numpy as np
import torch

from tqdm import tqdm

def prediction(data_loader, model, device, prediction_only=True):
    model.eval()
    preds = []

    if prediction_only == False:
        true_y = []
        for x, y in tqdm(data_loader):
            true_y.append(y)
            x = x.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        true_y = torch.cat(true_y, dim=0).numpy()
        return preds, true_y
    
    else:
        for x in tqdm(data_loader):
            x = x.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        return preds        

def evaluate(data_loader, model, device):
    preds, true_y = prediction(data_loader, model, device, prediction_only=False)
    mse = ((np.array(preds) - np.array(true_y)) ** 2).mean()
    return mse

def save_metrics(train_mse, valid_mse, file):
    with open(file, 'w') as f:
        json.dump({'train_mse': float(train_mse), 'valid_mse': float(valid_mse)}, f)

def save_pred(preds, file):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tested_positive'])
        for idx, pred in enumerate(preds):
            writer.writerow([idx, pred])