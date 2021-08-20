import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

target = 'scale'

# IP
plot_mode = 'all_in_one'
obj = 'occ'

# Port
flow_dir = 'all'
port_dir = 'sys'
user_plot_pr = ['TCP']
user_plot_pr = ['UDP']
port_hist = pd.DataFrame({'A' : []})
user_port_hist = pd.DataFrame({'A' : []})

def acf(x, length=10):
  return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
      for i in range(1, length)])

def scale_check(data_idx, plot=False):
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    names = ['stan_b', 'stan_a', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc' or files[data_idx] == 'stan':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    
    scale_list = []
    for col in ['byt', 'pkt']:
        scale_list.append(col)
        scale_list.append(str(np.min(df[col])))
        scale_list.append(str(np.log(np.max(df[col]))))
        scale_list.append(';')

    print(files[data_idx], ':', (' '.join(scale_list)))

def pr_distribution(data_idx, plot=False):
    files = ['stan','stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    names = ['stan_fwd','stan_b', 'stan_a', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc' or files[data_idx] == 'stan':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    
    # pr marginal distribution
    pr_series = df['pr'].value_counts()
    print(names[data_idx], pr_series)
    ct = [0, 0, 0]
    for i in pr_series.keys():
        if i == 'TCP':
            ct[0] += pr_series[i]
        elif i == 'UDP':
            ct[1] += pr_series[i]
        else:
            ct[2] += pr_series[i]
    ct2 = [x/sum(ct) for x in ct]
    print(ct2)

    with open('results/pr/pr_marginal.csv', 'a') as out:
        out.write(','.join([names[data_idx], str(ct2[0]), str(ct2[1]), str(ct2[2]), '\n']))

    # prob of spec ports
    # http 80/tcp
    # https 443/tcp, 443/udp
    # ssh 22/tcp
    # DNS Service 53
    # FTP 21/tcp
    
    # ob_ports = [80, 443, 22, 53, 21]
    # for ob_q in ob_ports:
    #     df_ = df[df['dp'] == ob_q]
    #     print(ob_q, len(df_.index)/len(df.index), len(df_.index), len(df.index))
    #     input()
    

def check_distribution(df, name, user=None):
    # count = df_all.value_counts()
    # df.hist = df.hist()
    df = df.astype(int)

    # print(df.value_counts(normalize=True))
    global port_hist
    global user_port_hist
    if port_dir == 'sys':
        df.hist(bins=1024)  # s is an instance of Series
        # plt.plot(df.value_counts().index, df.value_counts().values)
        plt.savefig('./results/ports/%s/%s.png' % (port_dir, name))
        plt.clf()
        port_hist[name+'_port'] = df.value_counts(normalize=True)[:10].index
        port_hist[name+'_occ'] = df.value_counts(normalize=True)[:10].values
    else:
        l_p = []
        l_o = []
        bar_size = 6000
        for i in range(1024, 65536, bar_size):
            l_p.append(i)
            l_o.append(len(df[(i<=df) & (df<i+bar_size)].index))
            
            # print(df[(i<=df) & (df<i+bar_size)])
            # print(i, i+bar_size)
            # input()
        # print(l_o, name)
        
        l_o = [x/sum(l_o) for x in l_o]
        if len(user_port_hist.columns) == 1:
            user_port_hist[name+'_port'] = l_p
        user_port_hist[name+'_occ'] = l_o
        plt.plot(l_p, l_o)
        plt.xlabel("user port")
        plt.ylabel("probability")
        plt.title("user port distribution")

        # plt.xticks(x_pos, x)
        # plt.savefig('./results/ports/%s/%s.png' % (port_dir, name))
        # plt.clf()
    
    print('plotted %s' % name)

def port_distribution(data_idx, plot=False):
    files = ['stan','stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    names = ['stan_fwd','stan_b', 'stan_a', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc' or files[data_idx] == 'stan':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    for pr in ['TCP', 'UDP']:
        df_pr = df[df['pr'] == pr]
        if flow_dir == 'outgoing':
            flows = df_pr[df_pr['sa'].str.startswith('42.219')]
        elif flow_dir == 'incoming':
            flows = df_pr[df_pr['da'].str.startswith('42.219')]
        else:
            flows = df_pr.dropna()

        # outgoing_port = pd.concat([outgoing_flows['sp'], outgoing_flows['dp']], axis= 0)
        # check_distribution(outgoing_port, files[data_idx]+'_outgoing')

        # incoming_port = pd.concat([flows['sp'], flows['dp']], axis= 0)
        if port_dir == 'sys':
            incoming_port = flows[flows['dp']<1024]['dp']
            check_distribution(incoming_port, names[data_idx]+'_'+ pr +'_'+flow_dir)
        else:
            user_port = flows[flows['dp']>=1024]['dp']
            check_distribution(user_port, names[data_idx]+'_'+ pr +'_'+flow_dir, user=True)

def attribute_autocorr(data_idx, plot=False):
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    # files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'real'] 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    df1 = df[['byt', 'pkt']]
    # print(df1)
    # input()
    auto = acf(df1['byt'])
    print(files[data_idx], auto)
    if plot:
        # df_plot = pd.read_csv('results/ip_power_law/volumn_%s.csv' % files[data_idx], header=None)
        # print(df_plot)
        # input()
        plt.plot(auto)
        # plt.plot(df_plot[1])
        if plot_mode != 'all_in_one':
            plt.savefig('results/ip_power_law/%s.png' % files[data_idx])
            plt.clf()

def ip_volumne(data_idx, plot=False):
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    # files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'real'] 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    df1 = df[['sa', 'da', 'byt']]
    df2_s = df1[['sa', 'byt']]
    df2_d = df1[['da', 'byt']]
    df2_s.columns = ['ip', 'byt']
    df2_d.columns = ['ip', 'byt']

    df_all = pd.concat([df2_s, df2_d], axis= 0)
    df_nolocal = df_all[~df_all['ip'].str.startswith('42.219')]
    df_nolocal = df_nolocal.sample(1000000)

    group_cols = df_nolocal.columns.tolist()
    group_cols.remove('byt')

    df_sum = df_nolocal.groupby(group_cols,as_index=False)['byt'].sum()
    
    # print(df_sum)
    # print(df_sum[df_sum['ip']=='185.165.153.56'])
    # input()
    count = df_sum['byt']#.value_counts()
    s = count.sort_values(ascending=False)
    # final_df = count.sort_values(by=['byt'], ascending=False)
    # print(final_df)
    # input()
    print(files[data_idx], len(df.index), len(df_nolocal.index), len(df_sum.index), len(count.index))
    
    s.to_csv('results/ip_power_law/volumn_%s.csv' % files[data_idx])

    if plot:
        df_plot = pd.read_csv('results/ip_power_law/volumn_%s.csv' % files[data_idx], header=None)
        # print(df_plot)
        # input()
        plt.plot(np.log(df_plot[1]))
        # plt.plot(df_plot[1])
        if plot_mode != 'all_in_one':
            plt.savefig('results/ip_power_law/%s.png' % files[data_idx])
            plt.clf()

def ip_power_law(data_idx, plot=False):
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real']
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'real'] 
    if files[data_idx] == 'real':
        df = pd.read_csv("./postprocessed_data/%s/day2_90user.csv" % files[data_idx])
    elif files[data_idx] == 'stanc':
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0)) 
    else:
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], 0), index_col=None) 
        li = [df]
        for piece_idx in range(1, 5):
            df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx), index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)
    
    # count = df['sa'].value_counts()
    # df = df.sample(1000000)
    
    df_all = pd.concat([df['sa'], df['da']], axis= 0)
    df_nolocal = df_all[~df_all.str.startswith('42.219')]
    df_nolocal = df_nolocal.sample(1000000)
    count = df_nolocal.value_counts()
    # count_no_local = count[~count.index.str.startswith('42.219')]
    if plot_mode == 'all_in_one':
        # count = count[:38000]
        pass
    # print(len(count[count.index.str.startswith('42.219')].index))
    print(len(count.index))
    count.to_csv('results/ip_power_law/%s.csv' % files[data_idx])

    if plot:
        df_plot = pd.read_csv('results/ip_power_law/%s.csv' % files[data_idx], header=None)
        # print(df_plot)
        plt.plot(np.log(df_plot[1]))
        if plot_mode != 'all_in_one':
            plt.savefig('results/ip_power_law/%s.png' % files[data_idx])
            plt.clf()
        # input()

if __name__ == "__main__":
    if target == 'scale':
        for i in range(7):
            scale_check(i)

    if target == 'ip':
        for i in range(7):
            if obj == 'occ':
                ip_power_law(i, True)
            elif obj == 'vol':
                ip_volumne(i, True)
            else:
                attribute_autocorr(i, True)
            
            if obj == 'occ' and i == 4:
                break
        if plot_mode == 'all_in_one':
            if obj == 'occ':
                plt.legend(['STAN-b', 'STAN-a', 'B3', 'B4', 'Real'])
                plt.ylabel('occurence records (ln scale)')
                plt.xlabel('unique IP address')
            else:
                plt.legend(['STAN-b', 'STAN-a', 'B3', 'B4', 'B1', 'B2', 'Real'])
                plt.ylabel('total byt volume (ln scale)')
                plt.xlabel('unique IP address')
            if obj == 'vol' or obj == 'occ':
                plt.xscale('log')
            plt.savefig('results/ip_power_law/%s.png' % obj)


    elif target == 'port':
        for i in range(8):
            port_distribution(i, True)
        global port_hist
        global user_port_hist
        if port_dir == 'sys':
            port_hist.to_csv('results/ports/sys/daonly_%s.csv' % flow_dir, index=False)
        else:
            user_port_hist.to_csv('results/ports/user/daonly_%s.csv' % flow_dir, index=False)
            plt.legend(['stan_b', 'stan_a', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real'])
            plt.savefig('./results/ports/%s/all.png' % (port_dir))    

    elif target == 'pr':
        for i in range(8):
            pr_distribution(i, True)
        
