import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
from sklearn.preprocessing import LabelEncoder
from IPython.display import display


import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, comp_data, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', season=None,
                 target='OT', scale=True, inverse=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.comp_data = comp_data
        self.season = season
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = self.comp_data

        if self.season == "pre-pandemic":
            df_raw = df_raw[df_raw['PENDS'] > '2011-03-31']
            df_raw = df_raw[df_raw['PENDS'] < '2020-01-01']
        elif self.season == "post-pandemic":
            df_raw = df_raw[df_raw['PENDS'] > '2020-01-01']

        df_raw.rename(columns={'PENDS': 'date'}, inplace=True)

        self.uniq_companies = df_raw["OFTIC"].unique()
        self.comp_ticker = df_raw["OFTIC"]
        df_raw.drop(columns=["Date", "Sector", "Unnamed: 0"], inplace=True)
        df_raw.drop(columns=["OFTIC"], inplace=True)


        assert not df_raw.isna().any().any(), "df_raw contains NaN values after initial processing"

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        # print(df_raw.select_dtypes(exclude=[np.number]))

        num_train = int(len(df_raw)*0.8)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0,  num_train-self.seq_len]
        border2s = [num_train, num_train+num_test]

        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        # border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]


        sector_col = 'numeric_sector'

        if sector_col in df_data.columns:
            sector_data = df_data[sector_col].values.reshape(-1, 1)
            data_to_scale = df_data.drop(columns=[sector_col])
        else:
            sector_data = None
            data_to_scale = df_data

        if self.scale:
            train_data = data_to_scale[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            scaled_data = self.scaler.transform(data_to_scale.values)

            if sector_data is not None:
                data = np.concatenate([scaled_data, sector_data], axis=1)
            else:
                data = scaled_data
        else:
            data = df_data.values

        # Ensure the data is in the correct order: ['date', ...(other features), sector_col, target]
        if sector_data is not None:
            columns_order = data_to_scale.columns.tolist() + [sector_col]
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order.remove(sector_col)
            columns_order =  columns_order + [sector_col] + [self.target]
            data = data[columns_order]
        else:
            columns_order = data_to_scale.columns.tolist()
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order =  columns_order + [self.target]
            data = data[columns_order]

        # display(data)
        assert not data.isna().any().any(), "data contains NaN values after initial processing"
        assert not df_data.isna().any().any(), "df_data contains NaN values after initial processing"
        data = data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        assert not np.isnan(self.data_y).any(), "data_y contains NaN values after initial processing"
        assert not np.isnan(self.data_x).any(), "data_x contains NaN values after initial processing"

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        # print(seq_x)
        # display(self.comp_ticker.iloc[s_begin:s_end])
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        assert not np.isnan(seq_x).any(), "seq_x contains NaN values"
        assert not np.isnan(seq_y).any(), "seq_y contains NaN values"
        assert not np.isnan(seq_x_mark).any(), "seq_x_mark contains NaN values"
        assert not np.isnan(seq_y_mark).any(), "seq_y_mark contains NaN values"

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_validation(Dataset):
    def __init__(self, comp_data, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', season=None,
                 target='OT', scale=True, inverse=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.comp_data = comp_data
        self.season = season
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = self.comp_data

        if self.season == "pre-pandemic":
            df_raw = df_raw[df_raw['PENDS'] > '2011-03-31']
            df_raw = df_raw[df_raw['PENDS'] < '2020-01-01']
        elif self.season == "post-pandemic":
            df_raw = df_raw[df_raw['PENDS'] > '2020-01-01']

        df_raw.rename(columns={'PENDS': 'date'}, inplace=True)

        self.uniq_companies = df_raw["OFTIC"].unique()
        self.comp_ticker = df_raw["OFTIC"]
        # df_raw.drop(columns=["Date", "Sector", "OFTIC", "Unnamed: 0"], inplace=True)
        df_raw.drop(columns=["OFTIC"], inplace=True)


        assert not df_raw.isna().any().any(), "df_raw contains NaN values after initial processing"

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        # print(df_raw.select_dtypes(exclude=[np.number]))

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]

        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        # border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]


        sector_col = 'numeric_sector'

        if sector_col in df_data.columns:
            sector_data = df_data[sector_col].values.reshape(-1, 1)
            data_to_scale = df_data.drop(columns=[sector_col])
        else:
            sector_data = None
            data_to_scale = df_data

        if self.scale:
            train_data = data_to_scale[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            scaled_data = self.scaler.transform(data_to_scale.values)

            if sector_data is not None:
                data = np.concatenate([scaled_data, sector_data], axis=1)
            else:
                data = scaled_data
        else:
            data = df_data.values

        # Ensure the data is in the correct order: ['date', ...(other features), sector_col, target]
        if sector_data is not None:
            columns_order = data_to_scale.columns.tolist() + [sector_col]
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order.remove(sector_col)
            columns_order =  columns_order + [sector_col] + [self.target]
            data = data[columns_order]
        else:
            columns_order = data_to_scale.columns.tolist()
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order =  columns_order + [self.target]
            data = data[columns_order]

        # display(data)
        assert not data.isna().any().any(), "data contains NaN values after initial processing"
        assert not df_data.isna().any().any(), "df_data contains NaN values after initial processing"
        data = data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        assert not np.isnan(self.data_y).any(), "data_y contains NaN values after initial processing"
        assert not np.isnan(self.data_x).any(), "data_x contains NaN values after initial processing"

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        # print(seq_x)
        # display(self.comp_ticker.iloc[s_begin:s_end])
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        assert not np.isnan(seq_x).any(), "seq_x contains NaN values"
        assert not np.isnan(seq_y).any(), "seq_y contains NaN values"
        assert not np.isnan(seq_x_mark).any(), "seq_x_mark contains NaN values"
        assert not np.isnan(seq_y_mark).any(), "seq_y_mark contains NaN values"

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# class Dataset_Custom(Dataset):
#     def __init__(self, comp_data, root_path, flag='train', size=None, 
#                  features='S', data_path='ETTh1.csv', season=None,
#                  target='OT', scale=True, inverse=True, timeenc=0, freq='h', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24*4*4
#             self.label_len = 24*4
#             self.pred_len = 24*4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train':0, 'val':1, 'test':2}
#         self.set_type = type_map[flag]
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols=cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.comp_data = comp_data
#         self.season = season
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()

#         df_raw = self.comp_data

#         if self.season == "pre-pandemic":
#             df_raw = df_raw[df_raw['PENDS'] > '2011-03-31']
#             df_raw = df_raw[df_raw['PENDS'] < '2020-01-01']

#         df_raw.rename(columns={'PENDS': 'date'}, inplace=True)

#         self.uniq_companies = df_raw["OFTIC"].unique()
#         self.comp_ticker = df_raw["OFTIC"]
#         df_raw.drop(columns=["Date", "Sector", "OFTIC", "Unnamed: 0"], inplace=True)

#         assert not df_raw.isna().any().any(), "df_raw contains NaN values after initial processing"

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         # cols = list(df_raw.columns); 
#         if self.cols:
#             cols=self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
#         df_raw = df_raw[['date']+cols+[self.target]]

#         # print(df_raw.select_dtypes(exclude=[np.number]))

#         num_train = int(len(df_raw)*0.7)
#         num_test = int(len(df_raw)*0.2)
#         num_vali = len(df_raw) - num_train - num_test

# #         border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
# #         border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]

#         border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
#         border2s = [num_train, num_train+num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
        
#         if self.features=='M' or self.features=='MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features=='S':
#             df_data = df_raw[[self.target]]


#         sector_col = 'numeric_sector'

#         if sector_col in df_data.columns:
#             sector_data = df_data[sector_col].values.reshape(-1, 1)
#             data_to_scale = df_data.drop(columns=[sector_col])
#         else:
#             sector_data = None
#             data_to_scale = df_data

#         if self.scale:
#             self.scaler.fit(data_to_scale.values)
#             scaled_data = self.scaler.transform(data_to_scale.values)

#             if sector_data is not None:
#                 data = np.concatenate([scaled_data, sector_data], axis=1)
#             else:
#                 data = scaled_data
#         else:
#             data = df_data.values

#         # Ensure the data is in the correct order: ['date', ...(other features), sector_col, target]
#         if sector_data is not None:
#             columns_order = data_to_scale.columns.tolist() + [sector_col]
#             data = pd.DataFrame(data, columns=columns_order)
#             columns_order.remove(self.target)
#             columns_order.remove(sector_col)
#             columns_order =  columns_order + [sector_col] + [self.target]
#             data = data[columns_order]
#         else:
#             columns_order = data_to_scale.columns.tolist()
#             data = pd.DataFrame(data, columns=columns_order)
#             columns_order.remove(self.target)
#             columns_order =  columns_order + [self.target]
#             data = data[columns_order]

#         # display(data)
#         assert not data.isna().any().any(), "data contains NaN values after initial processing"
#         assert not df_data.isna().any().any(), "df_data contains NaN values after initial processing"
#         data = data.values
            
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#         assert not np.isnan(self.data_y).any(), "data_y contains NaN values after initial processing"
#         assert not np.isnan(self.data_x).any(), "data_x contains NaN values after initial processing"

    
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len 
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]

#         # print(seq_x)
#         # display(self.comp_ticker.iloc[s_begin:s_end])
#         if self.inverse:
#             seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
#         else:
#             seq_y = self.data_y[r_begin:r_end]

#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         assert not np.isnan(seq_x).any(), "seq_x contains NaN values"
#         assert not np.isnan(seq_y).any(), "seq_y contains NaN values"
#         assert not np.isnan(seq_x_mark).any(), "seq_x_mark contains NaN values"
#         assert not np.isnan(seq_y_mark).any(), "seq_y_mark contains NaN values"

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len- self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]


        sector_col = 'numeric_sector'

        if sector_col in df_data.columns:
            sector_data = df_data[sector_col].values.reshape(-1, 1)
            data_to_scale = df_data.drop(columns=[sector_col])
        else:
            sector_data = None
            data_to_scale = df_data

        if self.scale:
            self.scaler.fit(data_to_scale.values)
            scaled_data = self.scaler.transform(data_to_scale.values)
            if sector_data is not None:
                data = np.concatenate([scaled_data, sector_data], axis=1)
            else:
                data = scaled_data
        else:
            data = df_data.values

        # Ensure the data is in the correct order: ['date', ...(other features), sector_col, target]
        if sector_data is not None:
            columns_order = data_to_scale.columns.tolist() + [sector_col]
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order.remove(sector_col)
            columns_order =  columns_order + [sector_col] + [self.target]
            data = data[columns_order]
        else:
            columns_order = data_to_scale.columns.tolist()
            data = pd.DataFrame(data, columns=columns_order)
            columns_order.remove(self.target)
            columns_order =  columns_order + [self.target]
            data = data[columns_order]

        data = data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


#class Dataset_EPS(Dataset):
    # def __init__(self, root_path, flag='train', size=None, 
    #              features='MS', data_path='Merged.csv', 
    #              target='EPS', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
    #     if size == None:
    #         self.seq_len = 24*4*4
    #         self.label_len = 24*4
    #         self.pred_len = 24*4
    #     else:
    #         self.seq_len = size[0]
    #         self.label_len = size[1]
    #         self.pred_len = size[2]
    #     # init
    #     assert flag in ['train', 'test', 'val']
    #     type_map = {'train':0, 'val':1, 'test':2}
    #     self.set_type = type_map[flag]
        
    #     self.features = features
    #     self.target = target
    #     self.scale = scale
    #     self.inverse = inverse
    #     self.timeenc = timeenc
    #     self.freq = freq
    #     self.cols=cols
    #     self.root_path = root_path
    #     self.data_path = data_path
    #     self.__read_data__()


    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     all_data = []
    #     for data_path in self.data_paths:
    #         df_raw = pd.read_csv(os.path.join(self.root_path, data_path))
    #         if self.cols:
    #             cols = self.cols.copy()
    #             cols.remove(self.target)
    #         else:
    #             cols = list(df_raw.columns)
    #             cols.remove(self.target)
    #             cols.remove('date')
    #         df_raw = df_raw[['date'] + cols + [self.target]]
    #         all_data.append(df_raw)

    #     all_data = pd.concat(all_data)

    #     num_train = int(len(all_data) * 0.7)
    #     num_val = int(len(all_data) * 0.1)
    #     num_test = len(all_data) - num_train - num_val

    #     borders = [0, num_train, num_train + num_val, len(all_data)]

    #     if self.features == 'M' or self.features == 'MS':
    #         cols_data = all_data.columns[1:]
    #         df_data = all_data[cols_data]
    #     elif self.features == 'S':
    #         df_data = all_data[[self.target]]

    #     if self.scale:
    #         train_data = df_data[:borders[1]]
    #         self.scaler.fit(train_data.values)
    #         data = self.scaler.transform(df_data.values)
    #     else:
    #         data = df_data.values

    #     df_stamp = all_data[['date']][:borders[-1]]
    #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
    #     data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

    #     self.data_x = data[:borders[-1]]
    #     if self.inverse:
    #         self.data_y = df_data.values[:borders[-1]]
    #     else:
    #         self.data_y = data[:borders[-1]]
    #     self.data_stamp = data_stamp