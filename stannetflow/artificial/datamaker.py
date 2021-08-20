from sklearn.mixture import GaussianMixture
from torch.distributions import Normal
import numpy as np
import torch
import torch.optim as optim
import logging
import pandas as pd
from argparse import ArgumentParser


class artificial_data_generator(object):
    def __init__(self, weight_list):
        self.weight_list = weight_list
        self.df_naive = None
        self.X ,self.y = None, None

    def sample(self, row_num=10000):
        self.df_naive = self._gen_continuous(row_num, self.weight_list)
        return self.df_naive
    
    def agg(self, agg=None):
        if agg is None:
            return None, None
        self.X, self.y = self._agg_window(self.df_naive, agg)
        return self.X, self.y

    def _gen_continuous(self, row_num, weight_list=[]):
        noise = Normal(0, 1)
        row_dep = weight_list[0]

        rt = []
        for i in range(row_num):
            samp = []
            if i == 0:
                samp.append(noise.sample().tolist())
            else:
                samp.append(row_dep*rt[-1][0] + (1-row_dep)*noise.sample().tolist())
            for col_dep_i in weight_list[1:]:
                samp.append(col_dep_i*samp[-1] + (1-col_dep_i)*noise.sample().tolist())
            rt.append(samp)

        df = pd.DataFrame.from_records(rt)
        return df

    def _agg_window(self, df_naive, agg_size):
        col_num = len(df_naive.columns)
        buffer = [[0]*col_num] * agg_size
        X, y = [], []

        list_naive = df_naive.values.tolist()
        for row in list_naive:
            buffer.append(row)
            row_with_window = []
            for r in buffer[-agg_size-1:]:
                row_with_window += r
            X.append(row_with_window)
            y.append(row)

        
        X = torch.Tensor(X).view(len(X), -1, col_num) #col_num*(agg_size+1)
        y = torch.Tensor(y).view(-1, col_num)
        
        return X, y