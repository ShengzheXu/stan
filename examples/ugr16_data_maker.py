import os
import sys
sys.path.insert(0, './../')

from models.synthesizer import TableFlowSynthesizer, TableFlowTransformer, CustomDatasetFromCSV
from models.network_traffic_transformer import NetworkTrafficTransformer
import numpy as np
import logging
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from argparse import ArgumentParser

def prepare_rawdata(args):
    count = 0
    ntt = NetworkTrafficTransformer()
    tft = TableFlowTransformer('data_ugr16/pr_last_training.csv')
    for f in glob.glob('data_ugr16/day1_data/*.csv'):
        print('making train for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        tft.push_back(df, agg=args.n_agg, transformer=ntt)
        count += 1
    print(count)

    count = 0
    tft = TableFlowTransformer('data_ugr16/pr_last_testing.csv')
    for f in glob.glob('data_ugr16/day2_data/*.csv'):
        print('making test for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        tft.push_back(df, agg=args.n_agg, transformer=ntt)
        count += 1
    print(count)

########################################################################
# training
########################################################################

def runner_train(args, train_file):
    print('='*25+'start loading train data'+'='*25)
    train_from_csv = CustomDatasetFromCSV(train_file, args.n_agg+1, args.n_col)
    train_loader = torch.utils.data.DataLoader(dataset=train_from_csv, batch_size=args.batch_size, shuffle=True, \
            num_workers=16, pin_memory=True)
    tfs = TableFlowSynthesizer(dim_in=args.n_col, dim_window=args.n_agg, 
            discrete_columns=[[11,12], [13, 14, 15]],
            categorical_columns={5:1670, 6:1670,
                7:256, 8:256, 9:256, 10:256},
            learning_mode=args.learning_mode,
            arch_mode='B'
            )
    
    tfs.batch_fit(train_loader, epochs=args.train_epochs)
    return tfs

def runner_validation(args, train_file, tfs=None):
    print('='*25+'start loading validation data'+'='*25)
    train_from_csv = CustomDatasetFromCSV(train_file, args.n_agg+1, args.n_col)
    train_loader = torch.utils.data.DataLoader(dataset=train_from_csv, batch_size=args.batch_size, shuffle=True, \
            num_workers=16, pin_memory=True)
    if tfs is None:
        tfs = TableFlowSynthesizer(dim_in=args.n_col, dim_window=args.n_agg, 
                discrete_columns=[[11,12], [13, 14, 15]],
                categorical_columns={5:1670, 6:1670,
                    7:256, 8:256, 9:256, 10:256},
                learning_mode=args.learning_mode,
                arch_mode='B'
                ) 
        tfs.load_model('ep76')
    tfs.validate_loss(train_loader)
    return tfs
    

def runner_sample(args, tfs=None, margin_file=None):
    if tfs is None:
        tfs = TableFlowSynthesizer(dim_in=args.n_col, dim_window=args.n_agg, 
                discrete_columns=[[11,12], [13, 14, 15]],
                categorical_columns={5:1670, 6:1670,
                    7:256, 8:256, 9:256, 10:256},
                learning_mode=args.learning_mode,
                arch_mode='B'
                ) 
        # tfs.load_model('ep76')
        checkpoint_map = {0:'ep60', 1:'epsep1', 2:'ep58', 3:'ep58', 4:'ep60',
                     5:'ep58', 6:'ep58', 7:'ep58', 8:'ep58', 9:'ep58', 10:'ep58',
                     11:'ep58', 13:'ep58'}
        tfs.load_model(checkpoint_map)
        # input()
    print('='*25+'start loading marginal data'+'='*25)
    margin_from_csv = CustomDatasetFromCSV(margin_file, args.n_agg+1, args.n_col)
    margin_loader = torch.utils.data.DataLoader(dataset=margin_from_csv, batch_size=1000, shuffle=True, \
            num_workers=16, pin_memory=True)
    
    user_to_gen = "42.219.145.151,42.219.152.127,42.219.152.238,42.219.152.246,42.219.153.113,42.219.153.115,42.219.153.140,42.219.153.146,42.219.153.154,42.219.153.158"
    user_to_gen = user_to_gen.split(',')
    all_users = ['42.219.145.151', '42.219.152.127', '42.219.152.238', '42.219.152.246', '42.219.153.113', '42.219.153.115', '42.219.153.140', '42.219.153.146', '42.219.153.154', '42.219.153.158', '42.219.153.159', '42.219.153.16', '42.219.153.165', '42.219.153.170', '42.219.153.174', '42.219.153.179', '42.219.153.187', '42.219.153.190', '42.219.153.193', '42.219.153.198', '42.219.153.210', '42.219.153.214', '42.219.153.216', '42.219.153.220', '42.219.153.221', '42.219.153.23', '42.219.153.238', '42.219.153.241', '42.219.153.246', '42.219.153.250', '42.219.153.35', '42.219.153.36', '42.219.153.45', '42.219.153.47', '42.219.153.5', '42.219.153.53', '42.219.153.59', '42.219.153.60', '42.219.153.71', '42.219.153.75', '42.219.153.80', '42.219.153.81', '42.219.153.82', '42.219.153.83', '42.219.153.9', '42.219.154.124', '42.219.154.134', '42.219.154.145', '42.219.154.152', '42.219.154.155', '42.219.154.18', '42.219.154.181', '42.219.154.184', '42.219.154.185', '42.219.154.189', '42.219.154.191', '42.219.155.115', '42.219.155.123', '42.219.155.128', '42.219.155.132', '42.219.155.19', '42.219.155.25', '42.219.155.27', '42.219.155.30', '42.219.155.68', '42.219.155.69', '42.219.155.72', '42.219.155.86', '42.219.155.87', '42.219.155.89', '42.219.155.91', '42.219.156.188', '42.219.156.190', '42.219.156.194', '42.219.156.227', '42.219.156.237', '42.219.156.240', '42.219.157.13', '42.219.157.220', '42.219.157.246', '42.219.157.28', '42.219.158.162', '42.219.158.163', '42.219.158.169', '42.219.158.205', '42.219.158.209', '42.219.158.211', '42.219.158.217', '42.219.158.223', '42.219.158.224']
    data_path = './sep4_ugr16_gen/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    ntt = NetworkTrafficTransformer()
    for user in all_users:
        if user in user_to_gen:
            continue
        print('generating for', user)
        samples = tfs.time_series_sample(86400, margin_loader)
        df = pd.DataFrame(samples.cpu().numpy())
        df_rev = ntt.rev_transfer(df, user)
        df_rev.to_csv(data_path+user+'_maskB.csv', index=False)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n-agg", type=int, default=5)
    argparser.add_argument("--n-col", type=int, default=16)
    argparser.add_argument("--batch-size", type=int, default=512)
    argparser.add_argument("--train-epochs", type=int, default=1000)
    argparser.add_argument("--loss-file", type=str, default='gen_ugr16/train_loss.csv')
    argparser.add_argument("--learning-mode", type=str, default='B')
    args = argparser.parse_args()
    
    # prepare_rawdata(args)

    # runner_train(args, 'data_ugr16/pr_last_training.csv')
    # runner_train(args, 'data_ugr16/pr_last_tinytrain.csv')
    # runner_validation(args, 'data_ugr16/pr_last_training.csv')
    runner_sample(args, margin_file='data_ugr16/pr_last_margin.csv')