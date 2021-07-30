# min-max scaling and Standardized [N(mu=0,sigma=1)]
# https://sebastianraschka.com/Articles/2014_about_feature_scaling.html

import pandas as pd
import numpy as np
import glob
import csv

memory_height = 5
memory_width = 16
gen_model = 2
train_user = 90
test_user = 1
data_path = './data_ugr16/'
output_data_path = './data_ugr16/tfn_'

class NetworkTrafficTransformer(object):
    def __init__(self):
        pass

    def _map_ip_str_to_int_list(self, ip_str, ipspace=None):
        ip_group = ip_str.split('.')
        label_rt = []
        rt = []
        pw = 1
        # print(ip_str)
        for i in list(reversed(range(len(ip_group)))):
            label_rt.append(int(ip_group[i]))
        for i in range(len(label_rt)):
            rt.append(label_rt[i]/ipspace)
        return rt, label_rt

    def _port_number_interpreter(self, port_num, portspace=None):
        rt = [port_num/portspace]

        def get_category(x):
            return (x-1024)//100+1024 if x >= 1024 else x
        label_rt = [get_category(port_num)]
        return rt, label_rt
    
    def rev_port(self, emb):
        pred_num = int(emb*1670)
        interv = (pred_num-1024) * 100 + 1024
        decode_port = pred_num if pred_num < 1024 else np.random.randint(interv, interv+100)
        if decode_port > 65535:
            decode_port = 65535
        return decode_port

    def rev_transfer(self, df, this_ip):
        bytmax = 20.12915933105231 # df['log_byt'].max()
        pktmax = 12.83
        tdmax = 363
        teTmax = 23 # df['teT'].max()
        teDeltamax = 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535
        td_max = 1430
        b_max = 20.12915933105231

        print(df.head()) 
        df['raw_scale_byt'] = np.exp(df[2]*b_max)
        df['raw_scale_pkt'] = np.exp(df[3]*pktmax)

        buffer = []
        for index, row in df.iterrows():
            line = [int(row[0]*24), row[1]*td_max, int(row['raw_scale_byt']), int(row['raw_scale_pkt']), row[4]*tdmax]
            if row[11] == 1:
                line.append(self.rev_port(row[5])) # sp
                line.append(self.rev_port(row[6])) # dp
                line.append(this_ip) #sa
                line.append('.'.join([str(int(da_i*256)) for da_i in row[7:7+4]])) #da
            else:
                line.append(self.rev_port(row[6])) # dp
                line.append(self.rev_port(row[5])) # sp
                line.append('.'.join([str(int(da_i*256)) for da_i in row[7:7+4]])) #da
                line.append(this_ip) #sa
            prt = ['TCP', 'UDP', 'Other']
            if row[13] == 1:
                line.append(prt[0])
            elif row[14] == 1:
                line.append(prt[1])
            else:
                line.append(prt[2])
            buffer.append(line)
        out_df = pd.DataFrame(buffer)
        out_df.columns = ['hour', 'time_delta', 'byt', 'pkt', 'time_duration', 'sp', 'dp', 'sa', 'da', 'pr']

        return out_df
   
    def transfer(self, df):
        df['log_byt'] = np.log(df['byt'])
        df['log_pkt'] = np.log(df['pkt'])
        bytmax = 20.12915933105231 # df['log_byt'].max()
        pktmax = 12.83
        tdmax = 363
        teTmax = 23 # df['teT'].max()
        teDeltamax = 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535
        td_max = 1430
        b_max = 20.12915933105231
        this_ip = df.iloc[0]['this_ip']
    
        buffer = []
        for index, row in df.iterrows():
            # each row: teT, delta_t, byt, in/out, tcp/udp/other, sa*4, da*4, sp_sig/sp_sys/sp_other, dp*3 
            line = [row['teT']/teTmax, row['teDelta']/td_max, row['log_byt']/b_max, row['log_pkt']/pktmax, row['td']/tdmax]
            label_line = [row['teT']/teTmax, row['teDelta']/td_max, row['log_byt']/b_max, row['log_pkt']/pktmax, row['td']/tdmax]
            # line = [row['teT']/teTmax, row['log_byt']/bytmax]
            # [out, in]
            sip_list, label_sip_list = self._map_ip_str_to_int_list(row['sa'], ipspace)
            dip_list, label_dip_list = self._map_ip_str_to_int_list(row['da'], ipspace)
            
            spo_list, label_spo_list = self._port_number_interpreter(row['sp'], portspace)
            dpo_list, label_dpo_list = self._port_number_interpreter(row['dp'], portspace)

            if row['sa'] == this_ip:
                #line += sip_list + dip_list
                line += spo_list 
                line += dpo_list + dip_list 
                line += [1, 0]
                
                label_line += label_spo_list 
                label_line += label_dpo_list + label_dip_list 
                label_line += [1, 0]
            else:
                line += dpo_list
                line += spo_list + sip_list
                line += [0, 1]

                label_line += label_dpo_list
                label_line += label_spo_list + label_sip_list
                label_line += [0, 1]

            line_pr = []
            if row['pr'] == 'TCP':
                line_pr = [1, 0, 0]
            elif row['pr'] == 'UDP':
                line_pr = [0, 1, 0]
            else:
                line_pr = [0, 0, 1]
            line += line_pr
            label_line += line_pr

            buffer.append(line)

        df = pd.DataFrame(buffer)
        # print(df)
        return df

if __name__ == "__main__":
    print('========================start makedata==================')
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    with open('input_data/all_train.csv','w') as f:
       pass
    with open('input_data/day2_test.csv', 'w') as f:
       pass
    count = 0
    tot_train_line = 0
    tot_test_line = 0
    train_files = []
    test_files = []
    tedelta_max = -1
    log_byt_max = -1
    for two_set in ['input_data/train_set', 'input_data/day2_data']:
        for f in glob.glob(two_set+'/*.csv'):
            df = pd.read_csv(f)
            tedelta_max = max(tedelta_max, df['teDelta'].max())
            log_byt_max = max(log_byt_max, np.log(df['byt'].max()))

    count = 0
    for f in glob.glob('input_data/train_set/*.csv'):
        if count == train_user:
            break
        print('making train for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        train_files.append(f)
        tot_train_line += transform_1_df(df, this_ip, 'input_data/all_train.csv', tedelta_max, log_byt_max)
        count += 1
    '''
    count = 0
    for f in glob.glob('input_data/day2_data/*.csv'):
        if count == test_user:
            break
        print('making test for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        test_files.append(f)
        tot_test_line += transform_1_df(df, this_ip, 'input_data/day2_test.csv', tedelta_max, log_byt_max)
        count += 1
    '''
    print('tot_train_line:', tot_train_line, 'tot_test_line', tot_test_line)
    print('td_max', tedelta_max, 'b_max', log_byt_max)
    
    with open('input_data/makedata_record.txt', 'w') as f:
        f.write("train set\n"+"\n".join(train_files))
        f.write("\ntest set\n"+"\n".join(test_files))
        f.write('\ntot_train_line: '+ str(tot_train_line) + ' tot_test_line:' + str(tot_test_line) + '\n')
        f.write('\ntd_max: '+ str(tedelta_max) + ' b_max: ' + str(log_byt_max) + '\n')

#TODO: normalization to fixed range
#TODO: append the autoencoder module
