from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Custom_validation
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
import pandas as pd
import plotly.express as px

from utils.tools import EarlyStopping, adjust_learning_rate, custom_time_series_folds, find_supported_splits
from utils.metrics import metric

import numpy as np

from torch.utils.data import ConcatDataset

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna

from sklearn.model_selection import TimeSeriesSplit

import os
import time

import warnings
import sys
from IPython.display import display
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args): 
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def init_data(self):
        args = self.args

        all_comp_data = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))

        if args.sector == "financials":
            all_comp_data = all_comp_data[all_comp_data["Sector"] == "Financials"]

        # all_comp_data = all_comp_data[~all_comp_data["OFTIC"].isin(["NVR"])]

        return all_comp_data


    def _get_data(self, flag, comp_data, validation=False):
        
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'Merged':Dataset_Custom,
            'MergedBasis':Dataset_Custom,
            'MergedFinancials':Dataset_Custom,
            'EPS': Dataset_Custom,
            'EPS_validation': Dataset_Custom_validation
        }

        if validation:
            Data = data_dict['EPS_validation']
        else:
            Data = data_dict[self.args.data]

        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq

        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred

        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            comp_data = comp_data,
            season = args.season,
            #data_path_company=args.data_path_company,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )

        #data_set1 = data_set[data_set['OFTIC'] == company]
        #data_set1.drop(columns=['OFTIC'],inplace=True)

        #print(data_set)
        #import sys
        #sys.exit(1)

        # print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False,
            )

        return data_set, data_loader
        

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
            
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):

            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss


    def tscv(self, setting, trial):
        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # time_now = time.time()
        
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=self.args.factor_lr, patience=self.args.patience_lr, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            all_comp_data = self.init_data()
            uniq_companies = all_comp_data["OFTIC"].unique()

            iter_count = 0
            train_loss = []
            vali_losses = []
            test_losses = []
            
            self.model.train()
            epoch_time = time.time()

            for ticker in uniq_companies:
                comp_data = all_comp_data[all_comp_data['OFTIC'] == ticker]
                comp_data = comp_data.sort_values(by='PENDS')

                assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

                train_data, train_loader = self._get_data(flag='train', comp_data=comp_data, validation=True)
                # vali_data, vali_loader = self._get_data(flag='val', comp_data=comp_data)
                # test_data, test_loader = self._get_data(flag='test', comp_data=comp_data)

                train_steps = len(train_loader)

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    assert not torch.isnan(batch_x).any()

                    
                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                    # print(pred)
                    # print(true)
                    loss = criterion(pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 50==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            for ticker in uniq_companies:
                comp_data_vali = all_comp_data[all_comp_data['OFTIC'] == ticker]
                comp_data_vali = comp_data_vali.sort_values(by='PENDS')

                assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

                vali_data, vali_loader = self._get_data(flag='val', comp_data=comp_data_vali, validation=True)
                test_data, test_loader = self._get_data(flag='test', comp_data=comp_data_vali, validation=True)

                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                vali_losses.append(vali_loss)
                test_losses.append(test_loss)

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = np.average(vali_losses)
            test_loss = np.average(test_losses)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            trial.report(vali_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # adjust_learning_rate(model_optim, epoch+1, self.args)
            scheduler.step(vali_loss)
        
        return vali_loss
    
    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=self.args.factor_lr, patience=self.args.patience_lr, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            all_comp_data = self.init_data()
            uniq_companies = all_comp_data["OFTIC"].unique()

            iter_count = 0
            train_loss = []
            vali_losses = []
            test_losses = []
            outlier_dict = {}
            
            self.model.train()
            epoch_time = time.time()

            for ticker in uniq_companies:
                comp_data = all_comp_data[all_comp_data['OFTIC'] == ticker]
                comp_data = comp_data.sort_values(by='PENDS')

                assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

                train_data, train_loader = self._get_data(flag='train', comp_data=comp_data)
                # vali_data, vali_loader = self._get_data(flag='val', comp_data=comp_data)
                # test_data, test_loader = self._get_data(flag='test', comp_data=comp_data)

                train_steps = len(train_loader)

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    assert not torch.isnan(batch_x).any()
                    
                    model_optim.zero_grad()
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                    # print(pred)
                    # print(true)
                    loss = criterion(pred, true)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 50==0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time()-time_now)/iter_count
                        left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            for ticker in uniq_companies:
                comp_data_vali = all_comp_data[all_comp_data['OFTIC'] == ticker]
                comp_data_vali = comp_data_vali.sort_values(by='PENDS')

                assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

                # vali_data, vali_loader = self._get_data(flag='val', comp_data=comp_data_vali)
                test_data, test_loader = self._get_data(flag='test', comp_data=comp_data_vali)

                # vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                # if vali_loss > 50:
                #     outlier_dict[ticker] = vali_loss

                # vali_losses.append(vali_loss)
                test_losses.append(test_loss)
            
            # for ticker, vali_loss in outlier_dict.items():
            #     print(f"{ticker}: {vali_loss}")

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = np.average(vali_losses)
            test_loss = np.average(test_losses)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, 0, test_loss))
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # adjust_learning_rate(model_optim, epoch+1, self.args)
            scheduler.step(train_loss)
            
        # best_model_path = path+'/'+'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    # def train(self, setting):

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()
        
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
    #     model_optim = self._select_optimizer()
    #     criterion =  self._select_criterion()

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()

    #     for epoch in range(self.args.train_epochs):

    #         all_comp_data = self.init_data()
    #         uniq_companies = all_comp_data["OFTIC"].unique()

    #         iter_count = 0
    #         train_loss = []
    #         vali_losses = []
    #         test_losses = []
            
    #         self.model.train()
    #         epoch_time = time.time()

    #         for ticker in uniq_companies:
    #             comp_data = all_comp_data[all_comp_data['OFTIC'] == ticker]
    #             comp_data = comp_data.sort_values(by='PENDS')

    #             assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

    #             train_data, train_loader = self._get_data(flag='train', comp_data=comp_data)
    #             vali_data, vali_loader = self._get_data(flag='val', comp_data=comp_data)
    #             test_data, test_loader = self._get_data(flag='test', comp_data=comp_data)

    #             train_steps = len(train_loader)

    #             for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
    #                 iter_count += 1
                    
    #                 model_optim.zero_grad()
    #                 pred, true = self._process_one_batch(
    #                     train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
    #                 loss = criterion(pred, true)
    #                 train_loss.append(loss.item())
                    
    #                 if (i+1) % 100==0:
    #                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                     speed = (time.time()-time_now)/iter_count
    #                     left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
    #                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                     iter_count = 0
    #                     time_now = time.time()
                    
    #                 if self.args.use_amp:
    #                     scaler.scale(loss).backward()
    #                     scaler.step(model_optim)
    #                     scaler.update()
    #                 else:
    #                     loss.backward()
    #                     model_optim.step()

    #         print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
    #         train_loss = np.average(train_loss)
    #         vali_loss = self.vali(vali_data, vali_loader, criterion)
    #         test_loss = self.vali(test_data, test_loader, criterion)

    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #         adjust_learning_rate(model_optim, epoch+1, self.args)
            
    #     best_model_path = path+'/'+'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))
        
    #     return self.model


    def test(self, setting):        
        self.model.eval()
        
        preds = []
        trues = []

        all_comp_data = self.init_data()
        uniq_companies = all_comp_data["OFTIC"].unique()

        for ticker in uniq_companies:
            comp_data = all_comp_data[all_comp_data['OFTIC'] == ticker]
            comp_data = comp_data.sort_values(by='PENDS')
                
            assert not comp_data.isna().any().any(), "DataFrame contains NaN values"

            test_data, test_loader = self._get_data(flag='test', comp_data=comp_data)
        
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
            
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        preds=preds.flatten()
        trues=trues.flatten()

        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}'.format(rmse,mape))
        print('mspe:{}, r2:{}'.format(mspe, r2))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        
        df = pd.DataFrame({'Index': range(len(preds)), 'True Values': trues, 'Predicted Values': preds})

        fig = px.line(df, x='Index', y=['True Values', 'Predicted Values'], labels={'value': 'Values', 'Index': 'Index'}, 
                    color_discrete_map={'True Values': 'blue', 'Predicted Values': 'red'})

        fig.update_layout(title='True vs Predicted Values', xaxis_title='Index', yaxis_title='Values')

        fig.show()

        oftic = "AAPL"
        indexes = []

        if self.args.season == "post-pandemic":
            oftic_index = np.where(uniq_companies == oftic)[0][0] * 3
            indexes = ['2023-Q1','2023-Q2','2023-Q3']
            df = pd.DataFrame({'True Values': trues[oftic_index:oftic_index+3], 'Predicted Values': preds[oftic_index:oftic_index+3]}, index=indexes)

        elif self.args.season == "pre-pandemic":
            oftic_index = np.where(uniq_companies == oftic)[0][0] * 7
            indexes = ["2018-Q2", "2018-Q3", "2018-Q4" ,"2019-Q1", '2019-Q2', '2019-Q3', '2019-Q4']
            df = pd.DataFrame({'True Values': trues[oftic_index:oftic_index+7], 'Predicted Values': preds[oftic_index:oftic_index+7]}, index=indexes)

        # Plotting
        fig = px.line(df, labels={'value': 'EPS', 'index': 'PENDS'}, color_discrete_map={'True Values': 'blue', 'Predicted Values': 'red'})
        fig.update_layout(title='AAPL EPS-PENDS', xaxis_title='PENDS', yaxis_title='EPS')
        fig.show()

        return mae, mse, r2



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # print(index)
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
                
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # print(outputs.shape)
        # print(batch_y)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            # batch_y = dataset_object.inverse_transform(batch_y)
        # print(f"after inverse train: {batch_y}")
            
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
