import argparse
import os
import torch

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import plotly.graph_objs as go
import plotly.offline as pyo
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from tqdm.notebook import tqdm

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informerstack',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='EPS', help='data')
parser.add_argument('--root_path', type=str, default='/home/arda/Senior Project/Senior_Project/Data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='longmerged_deneme_51.csv', help='data file')    
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='EPS', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--companies', type=int, default=276, help='num of unique companies.')
parser.add_argument('--season', type=str, default="pre-pandemic", help='pre pandemic or post pandemic season.')
parser.add_argument('--sector', type=str, default="all-sectors", help='all sectors or financial sectors')

parser.add_argument('--seq_len', type=int, default=4, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=1, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=57, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=57, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=4, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=4, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
parser.add_argument('--factor_lr', type=float, default=0.5, help='inverse output data')
parser.add_argument('--patience_lr', type=int, default=2, help='inverse output data')
parser.add_argument('--weight_decay', type=float, default=0.1, help='inverse output data')


parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

#print(args)

#import sysfloat
#sys.exit(1)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'Merged_Informer':{'data':'Merged_Informer.csv','T':'Y','M':[3,3,3],'S':[1,1,1],'MS':[3,3,1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

# def objective(trial):
#     config = {
#         "batch_size": trial.suggest_int("batch_size", 16, 96, step=16),
#         "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1),
#         "dropout": trial.suggest_float("dropout", 0.05, 0.30),
#         "train_epochs": trial.suggest_int("train_epochs", 5, 30),
#     }

#     args.weight_decay = config["weight_decay"]
#     args.dropout = config["dropout"]
#     args.batch_size = config["batch_size"]
#     args.train_epochs = config["train_epochs"]

#     exp = Exp_Informer(args)

#     setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
#             args.seq_len, args.label_len, args.pred_len,
#             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
#             args.embed, args.distil, args.mix, args.des, 0)

#     val_loss = exp.tscv(setting, trial)

#     return val_loss


# study_name = "Informer-Tuner"
# storage_url = "sqlite:///db.sqlite3"

# storage = optuna.storages.RDBStorage(url=storage_url)

# # Check if the study exists
# study_names = [study.study_name for study in optuna.study.get_all_study_summaries(storage=storage)]
# if study_name in study_names:
#     # Delete the study if it exists
#     print(f"Deleting study '{study_name}'")
#     optuna.delete_study(study_name=study_name, storage=storage_url)
# else:
#     print(f"Study '{study_name}' does not exist in the storage.")
    
# study = optuna.create_study(direction='minimize', 
#                             storage=storage_url, 
#                             sampler=TPESampler(),
#                             pruner=optuna.pruners.SuccessiveHalvingPruner(
#                             min_resource=2,  # Minimum amount of resource allocated to a trial
#                             reduction_factor=3,  # Reduction factor for pruning
#                             min_early_stopping_rate=2 # Minimum early-stopping rate
#                             ),
#                             study_name=study_name,
#                             load_if_exists=False)

# pbar = tqdm(total=20, desc='Optimizing', unit='trial')

# def callback(study, trial):
#     # Update the progress bar
#     pbar.update(1)
#     pbar.set_postfix_str(f"Best Value: {study.best_value:.4f}")

# study.optimize(objective, n_trials=20, callbacks=[callback])
# pbar.close()

# # Best hyperparameters
# print('Number of finished trials:', len(study.trials))
# print('Best trial:')
# trial = study.best_trial

# print('Value:', trial.value)
# print('Params:')
# for key, value in trial.params.items():
#     print(f'{key}: {value}')

# args.weight_decay = trial.params['weight_decay']
# args.dropout = trial.params['dropout']
# args.batch_size = trial.params['batch_size']
# args.train_epochs = trial.params['train_epochs']

Exp = Exp_Informer

scores_dict = {"mae": 0, "mse": 0, "r2": 0}

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.sector, args.season, args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, mse, r2 = exp.test(setting)

    scores_dict['mae'] += mae
    scores_dict["mse"] += mse
    scores_dict['r2'] += r2

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

print("Overall Results for 5 Experiment: ")
print(f"MAE: {scores_dict['mae']/(args.itr)}, MSE: {scores_dict['mse']/(args.itr)}, R2: {scores_dict['r2']/(args.itr)}")