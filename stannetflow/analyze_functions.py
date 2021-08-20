import pandas as pd
import numpy as np
from datetime import datetime
import configparser
import os
import sys

columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
theDate = '2016-04-11'
internal_ip = '42.219.'

normal_datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'
spam_datafile = 'D:\\research_local_data\\spam_april_week3_csv\\april\\week3\\spam_flows_cut.csv'

working_folder = './stan_data/ugr16/'
outputfile = working_folder+'raw_data/day_1_%s.csv'

def prepare_folders():
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    sub_folders = ['raw_data', 'cleaned_data', 'gen_data']
    for i in sub_folders:
        if not os.path.exists(working_folder+i+'/'):
            os.makedirs(working_folder+i+'/')

def do_write(df, filename):
    filename = outputfile % filename
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

def extract(theUserIP):
    chunkNum = 0
    gen_flag = True
    chunksize = 10 ** 6
    import gc

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        block_time1 = datetime.now()
        chunk = chunk[chunk['te'].str.startswith(theDate)]
        if (len(chunk.index) == 0):
            break

        chunkNum += 1
        # chunk = chunk.sample(n=int(len(chunk.index)/10),random_state=131,axis=0)
        if isinstance(theUserIP, list):
            for one_ip in theUserIP:
                chunk2 = chunk[(chunk['sa'] == one_ip) | (chunk['da'] == one_ip)]
                if gen_flag:
                    print(len(chunk2.index), "to write for", one_ip)
                    do_write(chunk2, one_ip)
                del chunk2
        else:
            chunk2 = chunk[chunk['sa'] == theUserIP]
            print(len(chunk2.index), "to write")
            do_write(chunk2, theUserIP)
            
        block_time2 = datetime.now()
        print("blockNum", chunkNum, ",time:", (block_time2-block_time1).seconds)
        del chunk
        gc.collect()

def sample_choice(filename, num_of_row):
    all_record = pd.read_csv(filename)
    all_record = all_record.sample(n=num_of_row,random_state=131,axis=0)
    do_write(all_record, 'sampled_10IPs')

occur_dict = {}
outgoing_dict = {}
incoming_dict = {}
def cal_stats(df):
    for r in zip(df['sa'], df['da']):
        # count occurence of the sa
        if r[0].startswith(internal_ip):
            if r[0] in occur_dict:
                occur_dict[r[0]] += 1
            else:
                occur_dict[r[0]] = 1
            # judge the outgoing traffic to an external ip
            if not r[1].startswith(internal_ip):
                if r[0] in outgoing_dict:
                    outgoing_dict[r[0]] += 1
                else:
                    outgoing_dict[r[0]] = 1
        # count occurence of the da
        if r[1].startswith(internal_ip):
            if r[1] in occur_dict:
                occur_dict[r[1]] += 1
            else:
                occur_dict[r[1]] = 1
            # judge the incoming traffic from an external ip
            if not r[0].startswith(internal_ip):
                if r[1] in incoming_dict:
                    incoming_dict[r[1]] += 1
                else:
                    incoming_dict[r[1]] = 1

def analyze():
    starttime = datetime.now()
    chunksize = 10 ** 6
    chunkNum = 0
    import gc

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        chunk = chunk[chunk['te'].str.startswith(theDate)]
        if (len(chunk.index) == 0):
            break
        chunk = chunk[chunk['sa'].str.startswith(internal_ip) | chunk['da'].str.startswith(internal_ip)]
        chunkNum += 1
        cal_stats(chunk)
        print("blockNum", chunkNum, "with", len(chunk.index))
        del chunk
        gc.collect()

    most_occure = sorted( ((v,k) for k,v in occur_dict.items()), reverse=True)
    print("most bi-direction traffic users", most_occure[:20])

    categories_count = [0, 0, 0, 0] # both sent&recieve, only sent to ex, only recieve from ex, interal-to-internal
    with open('data_stats.csv', 'w') as f:
        outstring = 'number_of_rows,ip,incoming_num,outgoing_num,user_type\n'
        for case in most_occure:
            if case[1] not in incoming_dict:
                incoming_dict[case[1]] = 0
            if case[1] not in outgoing_dict:
                outgoing_dict[case[1]] = 0
            user_type = -1
            if incoming_dict[case[1]]>0 and outgoing_dict[case[1]]>0:
                categories_count[0] += 1
                user_type = 0
            elif incoming_dict[case[1]]==0 and outgoing_dict[case[1]]>0:
                categories_count[1] += 1
                user_type = 1
            elif incoming_dict[case[1]]>0 and outgoing_dict[case[1]]==0:
                categories_count[2] += 1
                user_type = 2
            else: # incoming_dict[case[1]]==0 and outgoing_dict[case[1]]==0
                categories_count[3] += 1
                user_type = 3
            outstring += ','.join([str(case[0]), case[1], str(incoming_dict[case[1]]), str(outgoing_dict[case[1]]), str(user_type)]) + '\n'
        
        f.write(outstring)
    print("4 categories count: both sent&recieve, only sent to ex, only recieve from ex, interal-to-internal")
    print(categories_count)
    endtime = datetime.now()
    print('process time', (endtime-starttime).seconds)

def previous_check(previous_user_list, a_ip_list):
    retA = [i for i in previous_user_list if i in a_ip_list]
    print('previous ip type 0',len(retA), retA)


def plot_refer(stats_file, set_config_user=False, previous_user_list=None):
    a = pd.read_csv(stats_file)
    a = a[a['ip'].str.startswith(internal_ip)]
    a = a[a['user_type'] == 0]
    if previous_user_list is not None:
        previous_check(previous_user_list, a['ip'].tolist())
    sys.path.append('../')
    print("current sys path:", sys.path)
    from utils.plot_utils import plot_source_distribution

    num_of_connection = a['number_of_rows'].values.tolist()
    plot_source_distribution(np.log(num_of_connection))

    user_addresses = a['ip'].values.tolist()
    q_1 = int(len(num_of_connection)/4)
    median_index = int(len(num_of_connection)/2)
    q_3 = int(len(num_of_connection)*3/4)

    from random import sample
    selected_users_index = sample(range(q_1, q_3), 100)
    # selected_users_index = range(450, 551)
    
    for i in selected_users_index:
        print(user_addresses[i], num_of_connection[i])
    
    print('total:', len(num_of_connection), sum(num_of_connection))
    print('we selected between:', q_1, q_3, selected_users_index)
    
    selected_users = [str(user_addresses[x]) for x in selected_users_index]
    if set_config_user is True:
        config = configparser.ConfigParser()
        config.read('./ugr16_config.ini')
        config['DEFAULT']['userlist'] = ','.join(selected_users)
        config['GENERATE']['gen_users'] = ','.join(selected_users)
        
        with open('./ugr16_config.ini', 'w') as configfile:
            config.write(configfile)
    
def recover_userlist_from_folder():
    import glob
    source_folder = './../data/raw_data/'
    user_list = []
    how_long = 0
    for f in glob.glob(source_folder+'*.csv'):
        user_list.append(f.split('_')[-1][:-4])
        how_long += len(pd.read_csv(f).index)
    print('recovered userlist: %d rows in total\n' % how_long, user_list)
    config = configparser.ConfigParser()
    config.read('./ugr16_config.ini')
    config['DEFAULT']['userlist'] = ','.join(user_list)
    config['GENERATE']['gen_users'] = ','.join(user_list)
    
    with open('./ugr16_config.ini', 'w') as configfile:
        config.write(configfile)

# call this function with python3 UGR16.py [arg], where [arg] is '-a', '-p' or '-e' (for probe and extract seperately).
if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    config = configparser.ConfigParser()
    config.read('./ugr16_config.ini')
    normal_datafile = config['DEFAULT']['huge_data_path']
    previous_user_list = config['DEFAULT']['userlist'].split(',')

    if '-a' in sys.argv:
        print('reach analyze')
        analyze()
    
    if '-r' in sys.argv:
        print('recovering user list from folder')
        recover_userlist_from_folder()

    if '-p' in sys.argv or '-pw' in sys.argv:
        print('reach plot_stats')
        stats_file = 'bidirection_data_stats_3traffic.csv'
        stats_file = 'data_stats.csv'
        if '-pw' in sys.argv:
            plot_refer(stats_file, True, previous_user_list)
        else:
            plot_refer(stats_file,False, previous_user_list)
    
    if '-e' in sys.argv:
        print('reach extract')

        config = configparser.ConfigParser()
        config.read('./ugr16_config.ini')
        user_list = config['DEFAULT']['userlist'].split(',')
        print('extracting:', user_list)
        prepare_folders()
        extract(user_list)

        # for ip in ten_ips:
        #     print('now extracting: %s' % ip)
        #     extract(ip)
        # sample_choice(outputfile % 'merged_10IPs', 637608) # 637608, '42.219.158.226'
        
        


# 10667863, '42.219.156.211'
# 10664359, '42.219.156.231'
# 5982859, '42.219.159.95'
# 3995830, '42.219.153.191'
# 2760867, '42.219.155.28'
# 2126619, '42.219.153.62'
# 2031099, '42.219.159.85'
# 1740711, '42.219.158.156'
# below 10 IPs to build gen_data_version1
# 1366940, '42.219.153.7'      
# 1342589, '42.219.153.89'     
# 1210025, '42.219.155.56'
# 1175793, '42.219.155.26'
# 1081080, '42.219.159.194'
# 1046866, '42.219.152.249'
# 944492, '42.219.159.82'
# 888781, '42.219.159.92'
# 771349, '42.219.159.94'
# 637608, '42.219.158.226'           <======== standard