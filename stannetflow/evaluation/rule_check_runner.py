import pandas as pd


#test 1
succeeded_one_ip_intern = 0
failed_one_ip_intern = 0
def checkOneIPIntern(srcIP,dstIP):
    global succeeded_one_ip_intern
    global failed_one_ip_intern
    if srcIP[:6] == "42.219" or dstIP[:6] == "42.219" or srcIP == "0.0.0.0" or dstIP == "255.255.255.255":
        succeeded_one_ip_intern += 1
    else:
        failed_one_ip_intern += 1

#test 2
succeeded_tcp_80 = 0
failed_tcp_80 = 0
def checkPort80TCP(proto,srcPt,dstPt):
    global succeeded_tcp_80
    global failed_tcp_80
    if srcPt == 80 or srcPt == 443 or dstPt == 80 or dstPt == 443:
        if proto == "TCP":
            succeeded_tcp_80 += 1
        else:
            failed_tcp_80 += 1

#test 3
succeeded_udp_53 = 0
failed_udp_53 = 0
def checkPort53UDP(proto,srcPt,dstPt):
    global succeeded_udp_53
    global failed_udp_53
    if srcPt in [53, 52] or dstPt in [52,53]:
        if proto == "UDP":
            succeeded_udp_53 += 1
        else:
            failed_udp_53 += 1

#test 4
succeeded_multicast = 0
failed_multicast = 0
def checkMultiBroadcast(srcIP,dstIP,row):
    global succeeded_multicast
    global failed_multicast
    ip1_1 = int( srcIP.split(".")[0] )
    ip1_4 = int( srcIP.split(".")[3] )

    ip2_1 = int( dstIP.split(".")[0] )
    ip2_4 = int( dstIP.split(".")[3] )

    if (ip2_1 > 223 or (ip2_1 == 192 and ip2_4 == 255)) and ip1_1 < 224 and not(ip1_4 == 192 and ip1_4 == 255):
        succeeded_multicast += 1
    elif ip1_1 > 223 or (ip1_4 == 192 and ip1_4 == 255):
        failed_multicast += 1

#test5
succeeded_netbios = 0
failed_netbios = 0
def checkNetbios(srcIP,dstIP,dstPt,proto):
    global succeeded_netbios
    global failed_netbios
    ip1_1 = int( srcIP.split(".")[0] )
    ip1_2 = int( srcIP.split(".")[1] )

    ip2_1 = int( dstIP.split(".")[0] )
    ip2_4 = int( dstIP.split(".")[3] )

    if dstPt == 137 or dstPt == 138:
        if ip1_1 == 42 and ip1_2 == 219 and proto == "UDP" and ip2_1 == 42 and ip2_4 == 255:
            succeeded_netbios += 1
        else:
            failed_netbios += 1

succeeded_dsn21_check1 = 0
failed_dsn21_check1 = 0
reserved_ip_set = ['0.0.0.0', '10.0.0.0', '100.64.0.0', '127.0.0.0', '169.254.0.0',
    '172.16.0.0', '192.0.0.0', '192.0.2.0', '192.88.99.0', '192.168.0.0', '198.18.0.0',
    '198.51.100.0', '203.0.113.0', '224.0.0.0', '240.0.0.0', '255.255.255.255']
def dsn21_check_ip(srcIP, dstIP):
    global succeeded_dsn21_check1
    global failed_dsn21_check1
    print(srcIP, dstIP)
    input()
    if srcIP in reserved_ip_set or dstIP in reserved_ip_set:
        failed_dsn21_check1 += 1
    else:
        succeeded_dsn21_check1 += 1
    # ip1s = [int(x) for x in srcIP.split(".")]
    # # print(dstIP)
    # if len(dstIP) < 8:
    #     return
    # ip2s = [int(x) for x in dstIP.split(".")]
    # for i in range(4):
    #     if ip1s[i] < 0 or ip1s[i] > 255:
    #         failed_dsn21_check1 += 1
    #         return
    #     if ip2s[i] < 0 or ip2s[i] > 255:
    #         failed_dsn21_check1 += 1
    #         return
    # if all(elem == 0 or elem == 255 for elem in ip1s):
    #     print(ip1s)
    #     input()
    #     failed_dsn21_check1 += 1
    #     return 
    # if all(elem == 0 or elem == 255 for elem in ip2s):
    #     print(ip2s)
    #     input()
    #     failed_dsn21_check1 += 1
    #     return 
    # succeeded_dsn21_check1 += 1 

#test6
succeeded_byte_packet = 0
failed_byte_packet = 0
import numpy as np
def checkRelationBytePackets(bytzes,packets):
    global succeeded_byte_packet
    global failed_byte_packet
    # possible_bin = 1.0/200/2
    # min_edge = int(np.exp((np.log(40)/20.12915933105231-possible_bin)*20.12915933105231))
    # print(min_edge)
    if bytzes >= packets * 41 and bytzes <= packets * 65536:
        succeeded_byte_packet += 1
    else:
        failed_byte_packet += 1

#test7
succeeded_dur_one_packet = 0
failed_dur_one_packet = 0
def checkDurationOnePacket(duration,packets):
    global succeeded_dur_one_packet
    global failed_dur_one_packet
    if packets <= 1:
        d = float(duration)
        if d < 1: # duration == "0.000" or duration == "0" or d == 0: 
            succeeded_dur_one_packet += 1
        else:
            failed_dur_one_packet += 1


def output_rst(data_name, test_name, true_count, false_count):
    tot_count = true_count + false_count
    with open('results/rule_check_results.txt','a') as f:
        #print(data_name, '='*10, file=f)
        if tot_count > 0:
            print(test_name ,'true', true_count,'total', tot_count,'percent', true_count/tot_count, file=f)
        else:
            print(test_name, 'no sample', file=f)

def test_one_piece(data_idx, piece_idx):
    files = ['stanc', 'arcnn_f90', 'wpgan', 'ctgan', 'bsl1', 'bsl2', 'real1', 'real2'] 
    if files[data_idx] == 'real1':
        df = pd.read_csv("./postprocessed_data/real/day1_90user.csv" )
    elif files[data_idx] == 'real2':
        df = pd.read_csv("./postprocessed_data/real/day2_90user.csv" )
    else:
    #     df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx))
        df = pd.read_csv("./postprocessed_data/%s/%s_piece%d.csv" % (files[data_idx], files[data_idx], piece_idx))
    print('checking', files[data_idx])
    for index, row in df.iterrows():
        #print(row['c1'], row['c2'])
        # done
        # checkOneIPIntern(row['sa'], row['da'])
        # checkPort80TCP(row['pr'],row['sp'],row['dp'])
        # checkPort53UDP(row['pr'],row['sp'],row['dp'])
        checkRelationBytePackets(row['byt'],row['pkt'])
        
        # ignore
        # checkDurationOnePacket(row['td'],row['pkt'])
        #checkNetbios(row['sa'], row['da'],row['dp'],row['pr'])

        # dsn21_check_ip(row['sa'], row['da'])

    data_name = '%s_piece%d' % (files[data_idx], piece_idx)
    with open('results/rule_check_results.txt','a') as f:
        print(data_name, '='*10, file=f)
    # output_rst(data_name, 'test1', succeeded_one_ip_intern, failed_one_ip_intern)
    # output_rst(data_name, 'test2', succeeded_tcp_80, failed_tcp_80)
    # output_rst(data_name, 'test3', succeeded_udp_53, failed_udp_53)
    # #output_rst(data_name, 'test4', succeeded_netbios, failed_netbios)
    output_rst(data_name, 'test5', succeeded_byte_packet, failed_byte_packet)
    # output_rst(data_name, 'test6', succeeded_dur_one_packet, failed_dur_one_packet)
    # output_rst(data_name, 'test_dsn1', succeeded_dsn21_check1, failed_dsn21_check1)

def reset():
    global succeeded_one_ip_intern
    global failed_one_ip_intern

    global succeeded_tcp_80
    global failed_tcp_80

    global succeeded_dsn21_check1
    global failed_dsn21_check1

    global succeeded_byte_packet
    global failed_byte_packet

    succeeded_byte_packet = 0
    failed_byte_packet = 0

    succeeded_one_ip_intern = 0
    failed_one_ip_intern = 0

    succeeded_tcp_80 =0
    failed_tcp_80=0

    succeeded_dsn21_check1=0
    failed_dsn21_check1=0

if __name__ == "__main__":
    for i in range(6, 8):
        reset()
        for j in range(1):
            test_one_piece(i,j)
