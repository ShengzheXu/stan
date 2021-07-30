import os
import sys
sys.path.insert(0, './../')

from sklearn.mixture import GaussianMixture
from torch.distributions import Normal
from models.our_nets import SingleTaskNet
from models.synthesizer import TableFlowSynthesizer
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
        self.X, self.y = self._agg_window(df_naive, agg)
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

        X = torch.Tensor(X).view(-1, col_num*(agg_size+1))
        y = torch.Tensor(y).view(-1, col_num)
        # print(X)
        # print(y)
        # input()
        return X, y


def our_trainer(args, X, y, learning_mode):
    sample_size, col_num = X.size()
    print('data shape', sample_size, col_num)
    tfs = TableFlowSynthesizer(dim_in=2, dim_window=1, learning_mode=learning_mode)
    X = X.view(-1, 2, 2)
    tfs.fit(X, y, epochs=args.n_iterations)
    samples = tfs.sample(args.n_gen_samples)
    samples = pd.DataFrame(samples.numpy())
    return samples

def baseline_gmm(args, df_naive):
    np.random.seed(1)
    gmm_samples = {}
    for col_i in df_naive.columns:
        model_i = GaussianMixture(n_components=1)
        model_i.fit(np.reshape(df_naive[col_i].tolist(), (-1, 1)))
        col_i_samples, _ = model_i.sample(args.n_gen_samples)
        col_i_samples = np.reshape(col_i_samples, (-1))
        gmm_samples[col_i] = col_i_samples
    gmm_samples = pd.DataFrame(gmm_samples)
    # print(gmm_samples)
    return gmm_samples

def baseline_ctgan(args, df_naive):
    from ctgan import CTGANSynthesizer

    ctgan = CTGANSynthesizer()
    ctgan.fit(df_naive)
    ctgan_samples = ctgan.sample(args.n_gen_samples)
    # print(ctgan_samples)
    return ctgan_samples



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=400)
    argparser.add_argument("--n-gen-samples", type=int, default=10000)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    ########################################################################
    # generating artificial raw data
    ########################################################################
    adg = artificial_data_generator(weight_list=[0.9, 0.9])
    df_naive = adg.sample(row_num=args.n_gen_samples)
    X, y = adg.agg(agg=1)

    ########################################################################
    # training model and generating synthesis data
    ########################################################################
    # tfs_A_samples = our_trainer(args, X, y, learning_mode='A')
    tfs_B_samples = our_trainer(args, X, y, learning_mode='B')
    input()
    ########################################################################
    # generating baseline data
    ########################################################################
    # b1_samples = baseline_gmm(args, df_naive)
    # b4_samples = baseline_ctgan(args, df_naive)

    ########################################################################
    # export data
    ########################################################################
    data_path = './data_artificial/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df_naive.to_csv(data_path+'artificial_raw.csv', index=False)
    tfs_A_samples.to_csv(data_path+'artificial_tfs_a_2.csv', index=False)
    tfs_B_samples.to_csv(data_path+'artificial_tfs_b.csv', index=False)
    # b1_samples.to_csv(data_path+'artificial_b1.csv', index=False)
    # b4_samples.to_csv(data_path+'artificial_b4.csv', index=False)
