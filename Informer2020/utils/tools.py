import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        
        # Handle zero standard deviation to avoid division by zero
        self.std[self.std == 0] = 1

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    
class MinMaxScaler:
    def __init__(self):
        self.min = 0.
        self.max = 1.
    
    def fit(self, data):
        self.min = data.min(0)
        self.max = data.max(0)
        
        # # Handle zero range to avoid division by zero
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1

    def transform(self, data):
        min_ = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_ = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        return (data - min_) / (max_ - min_)

    def inverse_transform(self, data):
        min_ = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_ = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        if data.shape[-1] != min_.shape[-1]:
            min_ = min_[-1:]
            max_ = max_[-1:]
        return data * (max_ - min_) + min_

    
def find_supported_splits(comp_size):
    supported_splits = []
    for n_splits in range(1, comp_size + 1):
        if comp_size % n_splits != 0:
            continue
        
        supported_splits.append(n_splits)
    return supported_splits


def custom_time_series_folds(data, n_splits, company_list):

    total_size = len(data)
    comp_size = total_size // len(company_list)
    comp_fold_size = comp_size//n_splits

    if comp_size % n_splits != 0:
        supported_splits = find_supported_splits(comp_size)
        print(supported_splits)
        print(f"fold_size: {comp_fold_size} comp_size: {comp_size}")
        raise ValueError("Fold size must be divisible by the number of companies.")

    accumulated_train_idx = []     

    for i in range(n_splits-1):
        current_fold_val_idx = []
        current_fold_train_idx = []

        for j in range(len(company_list)):

            start_idx = j * comp_size
            val_start_idx = start_idx + (i+1) * comp_fold_size 
        
            end_idx = val_start_idx + comp_fold_size
        
            current_comp_train_idx = list(range(start_idx, val_start_idx))
            current_fold_train_idx.extend(current_comp_train_idx)  
        
            val_idx = list(range(val_start_idx, end_idx))
            current_fold_val_idx.extend(val_idx)  

        print(f"train: {current_fold_train_idx}")
        print(f"test: {current_fold_val_idx}")
        yield current_fold_train_idx, current_fold_val_idx