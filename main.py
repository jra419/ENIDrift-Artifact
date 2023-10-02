import time
import sys
import itertools
from datetime import datetime
from iP2Vmain import *
from numpy import *
from pandas import *
from measure import *
from ENIDrift_main import *


settings = {
        'num_run': 1,
        'release_speed': sys.argv[3],
        'lamda': [0.1, 0.1],
        'delta': [0.05, 0.05],
        'incremental': True,
        'save': True,
        'vector': False,
        'my_limit': 200000,
        'sampling': sys.argv[2],
        'attack': sys.argv[1]
}

# path_packet = 'data//packets.csv'
# path_label = 'data//labels.npy'

# path_packet = '/mnt/data/datasets/kitsune/os-scan/b/os-scan.csv'
# path_label = '/mnt/data/datasets/kitsune/os-scan/b/os-scan-labels.npy'

path_packet = str(sys.argv[4])
path_label = str(sys.argv[5])

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
label = load(path_label)
packets = read_csv(path_packet)

sampling = int(settings['sampling'])
attack = settings['attack']

if vec:
    vector_packet = load(path_vector)

print("\n\n**********ENIDrift**********\n\n")

for i_run in range(num_run):
    ENIDrift = ENIDrift_train(lamda = lamd, delta=delt, incremental=incre)
    FE = increPacket2Vector_main(path = path_packet, incremental=incre, sampl=sampling)

    ENIDrift.loadpara()
    FE.loadpara()
    prediction = []
    num_released = 0

    start = time.time()

    cur_labels = []
    cur_pkt = 0
    labels_sampl = []

    for i_packet in range(len(label)):

        if i_packet%10000 == 0:
            print('[info] '+str(i_packet)+' processed...')

        if (i_packet+1) % sampling != 0:
            continue
        # else:
        #     print(i_packet)

        # if i_packet%1000 == 0:
        #     print('[info] '+str(i_packet)+' processed...')

        # if i_packet == 1314687:
        #     print("last", label[i_packet-1])
        #     print("current", label[i_packet])

        cur_labels.append(label[i_packet])
        cur_pkt += 1

        # print("label row 1", label[0])
        # print("label row 64", label[63])
        # print("i_packet", str(i_packet))
        # print("label", label[i_packet])
        # if label[i_packet] == 1:
        #     print("!!! " + str(i_packet))
        #     break


        packet_extracted = FE.iP2Vrun().reshape(1, -1)
        prediction.append(ENIDrift.predict(packet_extracted))

        # Release labels
        # if i_packet % release_speed == 0:
        if cur_pkt % release_speed == 0:
            # print("i_packet", i_packet)
            # print(cur_labels)
            # print(np.asarray(cur_labels).shape)
            ENIDrift.update(np.asarray(cur_labels))
            # ENIDrift.update(label[num_released:i_packet+1])
            # num_released = i_packet + 1
            cur_pkt = 0
            labels_sampl.append(cur_labels)
            cur_labels = []

    labels_sampl.append(cur_labels)
    stop = time.time()
    ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Cut the inital round (where no classifier is actually trained).
    prediction = prediction[release_speed:]
    # labels_sampl = labels_sampl[release_speed:]

    # if s:
    #     ENIDrift.save()
    #     if not vec:
    #         FE.save()
    print("[info] Time elapsed for round "+str(i_run)+": "+str(stop-start)+" seconds")
    save(str(attack) + "-sampl-" + str(sampling) + "-r-" + str(release_speed) + "-" + str(ts_datetime) + "-result_prediction.npy", prediction)
    overall(prediction, np.asarray(list(itertools.chain(*labels_sampl)))[release_speed:], attack, sampling, release_speed)
