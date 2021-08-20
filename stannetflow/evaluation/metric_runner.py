import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.metric_utils import gd
from src.metric_utils import JS
from sklearn.metrics.pairwise import euclidean_distances

def load_folder():
    col = 'td'
    cache = None
    for f in glob.glob('train_set/*.csv'):
        df = pd.read_csv(f)
        if cache is None:
            cache = df[col]
        else:
            cache = pd.concat([cache, df[col]], ignore_index=True)
    for f in glob.glob('test_set/*.csv'):
        df = pd.read_csv(f)
        cache = pd.concat([cache, df[col]], ignore_index=True)
    #cache = cache.apply(lambda x: np.log(x))
    print(cache.max())

def plot_distribution():
    d = cache.hist().get_figure()
    d.savefig('2%s.jpg'%col)

def pr_mod(x):
    if x == 'TCP':
        return 0
    elif x == 'UDP':
        return 1
    else:
        return 2

def scale_process(df):
    df['byt'] = df['byt'].apply(lambda x: np.log(x+1))
    df['pkt'] = df['pkt'].apply(lambda x: np.log(x+1))
    df['pr'] = df['pr'].apply(pr_mod)
    rmv = ['byt','pkt','td','sp','dp','pr']
    for col in df.columns:
        if col not in rmv:
            del df[col]
    return df

def calc_JS(col1, col2):
    pk = gd(col1)
    qk = gd(col2)
    #js_score = euclidean_distances(pk, qk)
    js_score = JS(pk, qk, KL=None)
    return js_score

def swap_localip(df):
    df1 = df[(df['sa_0'] == 42) & (df['sa_1'] == 219)]
    df2 = df[(df['sa_0'] != 42) | (df['sa_1'] != 219)]
    
    df2['sa_0'], df2['da_0'] = df2['da_0'], df2['sa_0']
    df2['sa_1'], df2['da_1'] = df2['da_1'], df2['sa_1']
    df2['sa_2'], df2['da_2'] = df2['da_2'], df2['sa_2']
    df2['sa_3'], df2['da_3'] = df2['da_3'], df2['sa_3']
    df2['sp'], df2['dp'] = df2['dp'], df2['sp']
    
    df = pd.concat([df1, df2], axis=0,sort=False)
    print(df.columns)
    df.columns = ['td', 'localPort', 'outPort', 'pr', 'pkt', 'byt', 'localIP_0', 'localIP_1', 'localIP_2', 'localIP_3',
       'outIP_0', 'outIP_1', 'outIP_2', 'outIP_3']
    print(df.columns)
    return df

def see_f_dist(gen_data, name_id, piece_i):
    # plot_names = ['stan' ,'ANDS', 'B1','B2','B3','B4']
    # plot_names = ['stan']
    plot_names = gen_data
    gen_data = gen_data[name_i]
    #task = 'task3_'
    files = ['./postprocessed_data/real/day2_90user.csv']
    gen_file = './postprocessed_data/%s/%s' % (gen_data, gen_data) + '_all.csv' #'_piece%d.csv' % piece_i
    gen_file = './postprocessed_data/%s/%s' % (gen_data, gen_data) + '_piece%d.csv' % piece_i
    files.append(gen_file)
    plot_folder = './results/aaai_metric_plots/'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    dfs = []
    for f in files:
        print('processing:', f)
        temp_df = scale_process(pd.read_csv(f))
        temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
        temp_df = temp_df.dropna()
        #temp_df = swap_localip(scale_process(pd.read_csv(folder+task+f)))
        dfs.append(temp_df)
        print(list(dfs[-1].columns))
        print(dfs[-1].head(2))
    #input()
    col_len = len(list(dfs[-1].columns))
    col_ith = 0
    js_day2_arcnn = []
    js_day1_arcnn = []
    js_day2_day1 = []
    js_day2_gendata = []

    #fig, ax = plt.subplots(4, 14, sharex='col', sharey='row')
    print('data size:', len(dfs[0].index), 'vs', len(dfs[1].index))
    #df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
    for col in list(dfs[0].columns):
        print('doing', col)
        if col == 'pr':
            print('day2_pr_dist')
            print(type(dfs[0][col].value_counts()))
            for val, cnt in dfs[0][col].value_counts().iteritems():
                print(gen_data, val, cnt)  
            print('stan_pr_dist')
            print(type(dfs[1][col].value_counts()))
            for val, cnt in dfs[1][col].value_counts().iteritems():
                print(gen_data, val, cnt)  
            #input()
        #js_day2_day1.append(calc_JS(dfs[0][col], dfs[1][col]))
        js_day2_gendata.append(calc_JS(dfs[0][col], dfs[1][col]))
        #js_day1_arcnn.append(calc_JS(dfs[1][col], dfs[5][col]))

        # continue
        if col != 'byt':
            continue
        for i in range(len(dfs)):
            try:
            #if True:
                # print('==>', i, len(dfs))
                # input()
                df = dfs[i]
                #df[col].plot.kde(label=files[i])
                alp = 0.75 if i <= 1 else 0.5
                # use this following one
                # print('=====>', i, df[col])
                if col == 'byt':
                    df[col].plot.hist(density=True, bins=304, alpha=alp, label=files[i])
                    plt.xlim(0, 20)
                    plt.ylim(0, 1)
                else:
                    df[col].plot.hist(normed=True, alpha=alp, label=files[i])
                #=====
                #df[col].plot.hist(bins=304, alpha=alp, label=files[i])
                #plt.legend(loc='upper right')
                # if i == 0:
                    # plt.title(col +' of real test data')
                # else:
                    # plt.title(col + ' of ' +plot_names[name_id])
                
                plt.savefig(plot_folder+plot_names[name_id]+'_'+col+'_'+str(i)+'.png')
                plt.clf()
                print('ploting', plot_names[name_id], ':', col_ith, 'of', col_len, ':',col)
            except Exception as e:
                print(e)
                print('cant plot', col, 'of',files[i])
            
        col_ith += 1
    # return
    # with open('results/__ggplot_rev_kl_score.txt', 'a') as f:
    #     nm = list(dfs[0].columns)
    #     for i in range(len(nm)):
    #         out_ = [nm[i], str(js_day2_gendata[i]), gen_data]
    #         print(','.join(out_), file=f)

    # return
    with open('results/__js_score.txt', 'a') as f:
        print(list(dfs[0].columns), file=f) 
        #print(js_day2_day1, file=f) 
        #print(js_day1_arcnn, file=f) 
        out_ = [gen_data, str(piece_i)] + [str(x) for x in js_day2_gendata]
        print(gen_data, ',',piece_i,',',js_day2_gendata, file=f)
#        fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(12,7))
#        ax = ax.ravel() 
#        # this method helps you to go from a 2x3 array coordinates to 
#        # 1x6 array, it will be helpful to use as below
#
#        for idx in range(3):
#            ax[idx].hist(dfs[idx], bins=12, alpha=0.5)
#            ax[idx].hist(df.iloc[:,idx+3], bins=12, alpha=0.5)
#            ax[idx].set_title(df.columns[idx]+' with '+df.columns[idx+3])
#            ax[idx].legend(loc='upper left')

if __name__ == "__main__":
    # names = ['stan', 'arcnn_f90', 'bsl1','bsl2','wpgan','ctgan']
    # names = ['arcnn_f90', 'bsl1','bsl2','wpgan','ctgan']
    # names = ['stanc']
    names = ['stan', 'stanc','arcnn_f90', 'bsl1','bsl2','wpgan','ctgan']
    for name_i in range(0, len(names)):
        for k in range(5):
            see_f_dist(names, name_i, k)
            break
