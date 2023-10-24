import numpy as np
import pandas as pd
import NegativePool
import VectorDict

# For .csv file, which is already extracted from .pcap files
# In this version, the network packet field should contain
# ['srcIP', 'dstIP', 'srcproto', 'dstproto', 'srcMAC', 'dstMAC', 'protocol', 'len']

class increPacket2Vector:
    def __init__(
            self, path, lr, n_epoch, limit=50000000, dim=100, mode='unigram-table', a=0.75,
            n_negative=5, sgd='adagrad', kind='input', max_size_np=1e8, sampl=1,
            attack_flows_path=''):

        # initiate parameters
        # self.n_processed = 0 # the number of packets that have been processed
        self.n_processed = sampl - 1 # the number of packets that have been processed
        self.n_epoch = n_epoch
        self.limit = limit # the limit of the packet number

        # initiate the vector dictionary and the negative sample pool
        self.vec_dict = VectorDict.vector_dict(dim, lr, sgd, kind)
        self.ne_pool = NegativePool.negative_pool(a, mode, max_size_np, n_negative)

        # load data
        self.load_data(path)
        self.sampl=sampl

        self.attack_flows = self.load_attack_flows(attack_flows_path)

    def load_attack_flows(self, path):
        with open(path) as fp:
            data = [list(map(str, line.strip().split(' '))) for line in fp]
        print(data)
        return data

    def load_data(self, p):

        self.packets = pd.read_csv(p, dtype=str)
        self.limit = self.packets.shape[0]

    def preproc_packet(self, p_idx):

        srcIP = self.packets['srcIP'][p_idx]
        dstIP = self.packets['dstIP'][p_idx]
        # print(p_idx)
        # print(srcIP)
        # print(dstIP)
        if srcIP < dstIP:
            flow_name = srcIP+dstIP
        else:
            flow_name = dstIP+srcIP
        # if p_idx == 1314687:
        #     srcproto = self.packets['srcproto'][p_idx]
        #     dstproto = self.packets['dstproto'][p_idx]
        #     srcMAC = self.packets['srcMAC'][p_idx]
        #     dstMAC = self.packets['dstMAC'][p_idx]
        #     protocol = self.packets['protocol'][p_idx]
        #     lEN = self.packets['len'][p_idx]
        #     ts = self.packets['timestamp'][p_idx]
        #     print(str(p_idx))
        #     print(f'{srcIP}, {dstIP}, {srcproto}, {dstproto}, {srcMAC}, {dstMAC}, {protocol},\
        #         {lEN}, {ts}')
        #     print()

        return [flow_name] + [srcIP, dstIP, self.packets['srcproto'][p_idx], self.packets['dstproto'][p_idx],
                self.packets['srcMAC'][p_idx], self.packets['dstMAC'][p_idx], self.packets['protocol'][p_idx],
                self.packets['len'][p_idx]]

    def proc_packet(self):

        # print("n_processed", self.n_processed)

        # preprocess and update vocabulary
        ext_packet = self.preproc_packet(self.n_processed)
        self.vec_dict.update(ext_packet)

        # Check if current packet is labeled as an attack
        cur_label = 0
        for flow in self.attack_flows:
            # print("ext_packet src ip: ", ext_packet[1])
            # print("ext_packet dst ip: ", ext_packet[2])
            if ext_packet[1] == flow[0] and ext_packet[2] == flow[1]:
                # print("ATTACK FOUND")
                # print("ext_packet src ip: ", ext_packet[1])
                # print("ext_packet dst ip: ", ext_packet[2])
                # print("flow src ip: ", flow[0])
                # print("flow dst ip: ", flow[1])
                cur_label = 1

        # train
        for target in ext_packet:
            for context in ext_packet:
                # target and context cannot be the same
                if target == context:
                    continue

                # select negative samples
                neg_samples = self.ne_pool.get()

                # gradient descendent
                for i in range(self.n_epoch):
                    self.vec_dict.gradient_descendent(target, context, neg_samples)

        self.ne_pool.update(ext_packet)
        self.n_processed += self.sampl

        return [self.vec_dict.get(ext_packet[0]), cur_label]

    def next_packet(self):

        if self.limit <= self.n_processed:
            print(str(self.n_processed)+" processed, iP2V: off")
            return []
        else:
            return self.proc_packet()

    def sav(self):

        self.vec_dict.save_vec()

    def loa(self):

        self.vec_dict.load_vec()
