import numpy as np
import torch

from torch.utils.data import random_split
from typing import List, Optional

def train_valid_split(data_set: np.ndarray, valid_ratio: float, seed: int) -> List[np.ndarray]:
    '''Split provided training data into training set and validation set'''

    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def select_feature(train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray, feature_selection: Optional[List[int]]) -> List[np.ndarray]:
    '''Selects useful features to perform regression'''

    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if feature_selection is None:
        feature_index = list(range(raw_x_train.shape[1]))
    else:
        feature_index = feature_selection

    return raw_x_train[:,feature_index], raw_x_valid[:,feature_index], raw_x_test[:,feature_index], y_train, y_valid
