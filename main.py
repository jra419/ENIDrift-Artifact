import time
import sys
import itertools
import csv
import argparse
from datetime import datetime
from iP2Vmain import *
from numpy import *
from pandas import *
from measure import *
from ENIDrift_main import *

parser = argparse.ArgumentParser(description="ENIDrift args.")
parser.add_argument("--attack", type=str)
parser.add_argument("--sampling", type=int)
parser.add_argument("--release_speed", type=int)
parser.add_argument("--pcap", type=str)
parser.add_argument("--attack_flows", type=str)
args = parser.parse_args()

settings = {
        'num_run': 1,
        'release_speed': args.release_speed,
        'lamda': [0.1, 0.1],
        'delta': [0.05, 0.05],
        'incremental': True,
        'save': True,
        'vector': False,
        'my_limit': 200000,
        'sampling': args.sampling,
        'attack': args.attack
}

path_packet = args.pcap
path_attack_flows = args.attack_flows

vec = settings['vector']
my_limit = settings['my_limit']


if vec:
    path_vector = '..//rwdids//.npy'
else:
    path_vector = '-1'

#######################################################

num_run = settings['num_run']
release_speed = int(settings['release_speed'])
lamd = settings['lamda']
delt = settings['delta']
incre = settings['incremental']
s = settings['save']
packets = read_csv(path_packet)

with open(path_packet, 'r') as p:
    reader = csv.reader(p)
    headers = list(reader)

sampling = int(settings['sampling'])
attack = settings['attack']

if vec:
    vector_packet = load(path_vector)

print("\n\n**********ENIDrift**********\n\n")

enidrift_eval = []

for i_run in range(num_run):
    ENIDrift = ENIDrift_train(lamda = lamd, delta=delt, incremental=incre)
    FE = increPacket2Vector_main(path = path_packet, incremental=incre, sampl=sampling,
                                 attack_flows=path_attack_flows)

    ENIDrift.loadpara()
    FE.loadpara()
    prediction = []
    flow_headers = []
    num_released = 0

    start = time.time()

    cur_labels = []
    cur_pkt = 0
    labels_sampl = []

    for i_packet in range(len(packets)):

        if i_packet%10000 == 0:
            print('[info] '+str(i_packet)+' processed...')

        if (i_packet+1) % sampling != 0:
            continue

        packet_extracted_tmp = FE.iP2Vrun()
        packet_extracted = packet_extracted_tmp[0].reshape(1, -1)
        prediction.append(ENIDrift.predict(packet_extracted))
        flow_headers.append([headers[i_packet+1][0], headers[i_packet+1][1],
                             headers[i_packet+1][2], headers[i_packet+1][3]])

        cur_label = packet_extracted_tmp[1]
        cur_labels.append(cur_label)
        cur_pkt += 1

        # Release labels
        if cur_pkt % release_speed == 0:
            ENIDrift.update(np.asarray(cur_labels))
            cur_pkt = 0
            labels_sampl.append(cur_labels)
            cur_labels = []

    labels_sampl.append(cur_labels)
    stop = time.time()
    ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Cut the inital round (where no classifier is actually trained).
    prediction = prediction[release_speed:]
    flow_headers = flow_headers[release_speed:]
    labels_sampl = np.asarray(list(itertools.chain(*labels_sampl)))[release_speed:]

    for i in range(len(labels_sampl)):
        print(labels_sampl[i])

    for i in range(len(labels_sampl)):
        enidrift_eval.append([flow_headers[i][0], flow_headers[i][1], flow_headers[i][2],
                              flow_headers[i][3], prediction[i][0], prediction[i][2],
                              prediction[i][1], int(labels_sampl[i])])

    print("[info] Time elapsed for round "+str(i_run)+": "+str(stop-start)+" seconds")
    overall(prediction, labels_sampl, enidrift_eval, attack, sampling, release_speed)
