import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def plot_data(x, y):
    plt.hist2d(x, y, bins=50, cmap=plt.cm.BuPu, range=np.array([(-1, 1), (-1, 1)]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.axis('off')

def save_plot(save_file):
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    # plt.margins(0,0)
    plt.xlabel('x', fontsize=50)
    plt.ylabel('x-1', fontsize=50)

    plt.tick_params(labelsize=50)
    plt.savefig(save_file, bbox_inches='tight', pad_inches = 0)
    # plt.savefig(save_file)
    plt.clf()

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def corr_plot(data_path='./stan_data/', plot=False, plot_axis='xy'):
    df_naive = pd.read_csv(data_path+'artificial_raw.csv')
    df_naive.columns = df_naive.columns.astype(int)
    df_naive[2] = df_naive[0].shift(1)
    df_naive = df_naive.dropna()
    
    tfs_A_samples = pd.read_csv(data_path+'artificial_tfs_a_2.csv')
    tfs_A_samples.columns = tfs_A_samples.columns.astype(int)
    tfs_A_samples[2] = tfs_A_samples[0].shift(1)
    tfs_A_samples = tfs_A_samples.dropna()

    tfs_B_samples = pd.read_csv(data_path+'artificial_tfs_b.csv')
    tfs_B_samples.columns = tfs_B_samples.columns.astype(int)
    tfs_B_samples[2] = tfs_B_samples[0].shift(1)
    tfs_B_samples = tfs_B_samples.dropna()

    b1_samples = pd.read_csv(data_path+'artificial_b1.csv')
    b1_samples.columns = b1_samples.columns.astype(int)
    b1_samples[2] = b1_samples[0].shift(1)
    b1_samples = b1_samples.dropna()

    # b4_samples = pd.read_csv(data_path+'artificial_b4.csv')
    # b4_samples.columns = b4_samples.columns.astype(int)
    # tfs_C_samples = pd.read_csv(data_path+'artificial_tfs_prior.csv')
    # tfs_C_samples.columns = tfs_C_samples.columns.astype(int)

    ########################################################################
    # plotting gen data
    ########################################################################
    to_cmp = 1 if plot_axis == 'xy' else 2

    print('#'*30 + 'corr_xy' + '#'*30)
    corr_raw = np.corrcoef(df_naive[0], df_naive[to_cmp])[0, 1]
    corr_tfs_A = np.corrcoef(tfs_A_samples[0], tfs_A_samples[to_cmp])[0, 1]
    corr_tfs_B = np.corrcoef(tfs_B_samples[0], tfs_B_samples[to_cmp])[0, 1]
    corr_b1 = np.corrcoef(b1_samples[0], b1_samples[to_cmp])[0, 1]
    # corr_b4 = np.corrcoef(b4_samples[0], b4_samples[to_cmp])[0, 1]
    print('xy_corr_raw', corr_raw)
    print('xy_corr_tfs_A', corr_tfs_A)
    print('xy_corr_tfs_B', corr_tfs_B)
    print('xy_corr_b1', corr_b1)
    # print('xy_corr_b4', corr_b4)

    print('#'*30 + 'autocorr_x' + '#'*30)
    print('x_autocorr_raw', estimated_autocorrelation(df_naive[0].to_numpy())[:5])
    print('x_autocorr_tfs_A', estimated_autocorrelation(tfs_A_samples[0].to_numpy())[:5])
    print('x_autocorr_tfs_B', estimated_autocorrelation(tfs_B_samples[0].to_numpy())[:5])
    print('x_autocorr_b1', estimated_autocorrelation(b1_samples[0].to_numpy())[:5])
    # print('x_autocorr_b4', estimated_autocorrelation(b4_samples[0].to_numpy())[:5])

    if plot:
        data_path = './plots_artificial/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        plt.figure(figsize=(12, 12))
        
        plot_data(df_naive[0].to_numpy(), df_naive[to_cmp].to_numpy())
        save_plot(data_path+'observed_%s.png'%plot_axis)

        plot_data(tfs_A_samples[0].to_numpy(), tfs_A_samples[to_cmp].to_numpy())
        save_plot(data_path+'tfs_a_%s.png'%plot_axis)

        plot_data(tfs_B_samples[0].to_numpy(), tfs_B_samples[to_cmp].to_numpy())
        save_plot(data_path+'tfs_b_%s.png'%plot_axis)

        plot_data(b1_samples[0].to_numpy(), b1_samples[to_cmp].to_numpy())
        save_plot(data_path+'b1_%s.png'%plot_axis)

        # plot_data(b4_samples[0].to_numpy(), b4_samples[to_cmp].to_numpy())
        # save_plot(data_path+'b4_%s.png'%eval_plot)

        # plot_data(tfs_C_samples[0].to_numpy(), tfs_C_samples[to_cmp].to_numpy())
        # save_plot(data_path+'tfs_c.png')
        

def mse_same_row(data_path='./stan_data/'):
    print('='*30 + 'eval_task_same_row' + '='*30)
    def mse_xy(data_name):
        df = pd.read_csv(data_path+'artificial_%s.csv' % data_name)
        df.columns = df.columns.astype(int) 
        df_raw = pd.read_csv(data_path+'artificial_raw.csv')
        df_raw.columns = df_raw.columns.astype(int) 

        x = np.reshape(df[0].tolist(), (-1, 1))
        y = np.reshape(df[1].tolist(), (-1, 1))
        x_true = np.reshape(df_raw[0].tolist(), (-1, 1))
        y_true = np.reshape(df_raw[1].tolist(), (-1, 1)) 
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_true)
        print('%s_train_mse' % data_name, mean_squared_error(y_true, y_pred))
        # print('real_train_rmse', mean_squared_error(real_y, pred_y, squared=False))
    mse_xy('raw')
    mse_xy('tfs_prior_2')
    mse_xy('tfs_a_2')
    mse_xy('tfs_b')
    mse_xy('b1')
    mse_xy('b4')

def mse_temporal(data_path='./stan_data/'):
    print('='*30 + 'eval_task_temporal' + '='*30)
    def mse_x(data_name):
        df = pd.read_csv(data_path+'artificial_%s.csv' % data_name)
        df.columns = df.columns.astype(int)
        df_len = len(df.columns)
        df[df_len] = df[0].shift(-1)
        df = df.dropna()

        df_raw = pd.read_csv(data_path+'artificial_raw.csv')
        df_raw.columns = df_raw.columns.astype(int) 
        df_raw[df_len] = df_raw[0].shift(-1)
        df_raw = df_raw.dropna()

        x = df[[0, 1]].to_numpy() #np.reshape(df['raw_x'].tolist(), (-1, 1))
        y = df[[2]].to_numpy() 
        x_true = df_raw[[0, 1]].to_numpy() #np.reshape(df['raw_x'].tolist(), (-1, 1))
        y_true = df_raw[[2]].to_numpy() 
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_true)
        print('%s_train_mse' % data_name, mean_squared_error(y_true, y_pred))
        # print('real_train_rmse', mean_squared_error(real_y, pred_y, squared=False))
    mse_x('raw')
    mse_x('tfs_a')
    mse_x('tfs_b')
    mse_x('b1')
    mse_x('b4')

if __name__ == "__main__":
    corr_plot(plot=True, plot_axis='xx1')
    corr_plot(plot=True, plot_axis='xy')
    mse_temporal()
    mse_same_row()