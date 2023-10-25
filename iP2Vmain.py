import pandas as pd
import increPacket2Vector
import numpy as np
import time

class increPacket2Vector_main():

    def __init__(self, path, incremental, sampl, attack_flows):
        self.path = path
        self.lr = 0.05
        self.epoch = 1
        self.limit = 300000
        self.dim = 200
        self.a = 0.75
        self.n_negative = 5
        self.max_size_np = 1e8
        self.sampl = sampl
        self.ip2v = increPacket2Vector.increPacket2Vector(
            self.path, self.lr, self.epoch, self.limit, self.dim, 'unigram-table',
            self.a, self.n_negative, 'adagrad', 'input', self.max_size_np, self.sampl,
            attack_flows)
        self.i = 0

    def iP2Vrun(self):

        start = time.time()
        # self.i = self.i + 1
        self.i = self.i + self.sampl
        vector = self.ip2v.next_packet()
        end = time.time()
        return [np.array(vector[0]), vector[1], end - start, vector[2]]

    def loadpara(self):

        self.ip2v.loa()

    def save(self):

        self.ip2v.sav()
