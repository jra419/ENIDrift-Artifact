import os
import time
import argparse
import sys
import csv
from pathlib import Path
from numpy import *
from pandas import *
from iP2Vmain import *
from ENIDrift_main import *
from measure import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='ENIDrift.')
    argparser.add_argument("--attack", type=str)
    argparser.add_argument("--sampling", type=int)
    argparser.add_argument("--release_speed", type=int)
    argparser.add_argument("--pcap", type=str)
    argparser.add_argument("--attack_flows", type=str)
    args = argparser.parse_args()

    settings = {
        'num_run': 1,
        'release_speed': args.release_speed,
        'lamda': [0.1, 0.1],
        'delta': [0.05, 0.05],
        'incremental': True,
        'save': True,
        'vector': False,
        'sampling': args.sampling,
        'attack': args.attack
    }

    path_packet = args.pcap
    path_attack_flows = args.attack_flows

    num_run = settings['num_run']
    release_speed = int(settings['release_speed'])
    lamd = settings['lamda']
    delt = settings['delta']
    incre = settings['incremental']
    s = settings['save']
    print(path_packet)
    packets = read_csv(path_packet)

    with open(path_packet, 'r') as p:
        reader = csv.reader(p)
        headers = list(reader)

    sampling = int(settings['sampling'])
    attack = settings['attack']

    enidrift_eval = []

    ENIDrift = ENIDrift_train(lamda = lamd, delta=delt, incremental=incre)
    FE = increPacket2Vector_main(path = path_packet, incremental=incre, sampl=sampling,
                                 attack_flows=path_attack_flows)

    ENIDrift.loadpara()
    FE.loadpara()
    prediction = []
    flow_headers = []
    num_released = 0

    cur_labels = []
    cur_pkt = 0
    labels_sampl = []

    print("Running ENIdrift", flush=True)

    old_time = 0
    new_time = 0

    pkt_cnt = 0
    pkt_bytes = 0

    start  = time.time()

    dt_fe_total = 0
    dt_ad_total = 0

    for i_packet in range(len(packets)):

        if i_packet%10000 == 0:
            print('[info] '+str(i_packet)+' processed...')

        if (i_packet+1) % sampling != 0:
            continue

        packet_extracted_tmp = FE.iP2Vrun()
        packet_extracted = packet_extracted_tmp[0].reshape(1, -1)
        prediction_tmp, dt_ad_1 = ENIDrift.predict(packet_extracted)
        prediction.append(prediction_tmp)
        flow_headers.append([headers[i_packet+1][0], headers[i_packet+1][1],
                             headers[i_packet+1][2], headers[i_packet+1][3]])

        cur_label = packet_extracted_tmp[1]
        cur_labels.append(cur_label)
        cur_pkt += 1

        dt_fe = packet_extracted_tmp[2]
        framelen = packet_extracted_tmp[3]

        # Release labels
        if cur_pkt % release_speed == 0:
            print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            dt_ad_2 = ENIDrift.update(np.asarray(cur_labels))
            cur_pkt = 0
            labels_sampl.append(cur_labels)
            cur_labels = []
            dt_ad_total += dt_ad_2

        pkt_cnt += 1
        pkt_bytes += int(framelen)

        dt_fe_total += dt_fe
        dt_ad_total += dt_ad_1

    labels_sampl.append(cur_labels)
    stop = time.time()
    ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    runtime             = stop - start
    processing_rate_pps = int(pkt_cnt / runtime)
    processing_rate_bps = int((pkt_bytes * 8) / runtime)

    print(f'Total runtime    : {runtime:8.3f} s')
    print(f'FE time          : {dt_fe_total:8.3f} s ({100.0 * dt_fe_total/runtime:5.2f}%)')
    print(f'AD time          : {dt_ad_total:8.3f} s ({100.0 * dt_ad_total/runtime:5.2f}%)')
    print(f'Processing rate  : {processing_rate_pps} pps')
    print(f'                   {processing_rate_bps} bps')

    OUT_FILE = f'{args.attack}.csv'
    with open(OUT_FILE, 'w') as f:
        f.write(f'{runtime},')
        f.write(f'{dt_fe_total},')
        f.write(f'{dt_ad_total},')
        f.write(f'{processing_rate_pps},')
        f.write(f'{processing_rate_bps}\n')
