import numpy as np
from structured_data.nodes import Node, GroupOfNodes
from structured_data.utils import shilouette_score_list_of_group_of_nodes, dst

class Sequence:
    def __init__(self, name, values_to_sort, x, samples=None, inverse=False):
        self.n = len(values_to_sort)
        if samples is None:
            samples = [None] * self.n

        assert x.shape[0] == len(values_to_sort), "Length DOES NOT match!"
        assert self.n == len(samples), "Length of samples DOES NOT match!"
        self.name = name
        self.order = [val for val in values_to_sort]
        self.order.sort(key=None, reverse=inverse)
        self.value_to_index = {}
        for i, val in enumerate(self.order):
            self.value_to_index[val] = self.order.index(val)

        self.x = []
        self.samples = []
        for val in self.order:
            i = self.value_to_index[val]      
            self.x.append(x[i, :])
            self.samples.append(samples[i])

        self.x = np.array(self.x)
        self.current_idx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= self.n:
            self.current_idx = 0
            raise StopIteration()
        val = self.__getitem__(self.current_idx)
        self.current_idx += 1
        return val

    def __len__(self):
        return self.n
    
    def __add__(self, sequence):
        return Sequence(
            self.name, 
            self.order + sequence.order, 
            np.concatenate([self.x, sequence.x], axis=0),
            self.samples + sequence.samples
        )
    
    def __getitem__(self, i):
        return self.x[i ,:]
    
    def invert(self):
        return Sequence(self.name, self.order, self.x, self.samples, inverse=True)
    
    def compute_direction(self):
        direction = []
        for i in range(self.n):
            xx = [self.x[i + ii ,:] for ii in range(-1, 2) if 0 <= i + ii < self.n]
            connections = [xx[i + 1] - xx[i] for i in range(len(xx) - 1)]
            direction.append(np.mean(np.array(connections), axis=0))
        return direction
    
    def get_x(self):
        return self.x
    
    def cut(self, low, high):
        return Sequence(self.name, self.order[low:high], self.x[low:high, :], self.samples[low:high])
    
    def split(self, i):
        low = Sequence(self.name, self.order[:i], self.x[:i, :], self.samples[:i])
        high = Sequence(self.name, self.order[i:], self.x[i:, :], self.samples[i:])
        return low, high
    
    def to_grp_of_nodes(self):
        return GroupOfNodes([Node(self.order[i], self.x[i, :], self.samples[i]) for i in range(self.n)])
        
    def score_sample(self, i, th_close=1, th_far=[1, 4]):
        th_close = 1
        
        x_c = self.x[i, :]

        distances = dst(x_c, self.x)
        dst_close = []
        dst_far = []
        
        for j in range(self.n):
            if i == j:
                continue
            elif np.abs(i - j) <= th_close: 
                dst_close.append(distances[j])
            elif th_far[0] < np.abs(i - j) <= th_far[1]:
                dst_far.append(distances[j])

        dst_close = np.mean(dst_close)
        dst_far = np.mean(dst_far)
        return dst_close, dst_far

    def score_representation(self):
        dst_close, dst_far = [], []
        for i in range(self.n):
            dst_1, dst_2 = self.score_sample(i)
            dst_close.append(dst_1)
            dst_far.append(dst_2)
        dst_close = np.mean(dst_close)
        dst_far = np.mean(dst_far)
        score = (dst_far - dst_close) / dst_far
        return score
    
    def match_sequence(self, seq, match_window):
        ind_1, ind_2 = None, None
        match_err = np.inf
        for i1 in range(self.n):
            i1_end = i1+match_window
            if i1_end >= self.n-1:
                continue
            x_1 = self.x[i1:i1_end, :]
            for i2 in range(seq.n):
                i2_end = i2+match_window
                if i2_end >= seq.n-1:
                    continue
                x_2 = seq.x[i2:i2_end, :]
                err = np.sqrt(np.sum(np.square(x_1 - x_2)))
                if match_err > err:
                    match_err = err
                    ind_1, ind_2 = i1, i2
        return ind_1, ind_2, match_err


class Sequences:
    def __init__(self, list_of_sequences):
        self.keys = []
        self.seqs = {}
        for seq in list_of_sequences:
            if seq.name not in self.keys:
                self.keys.append(seq.name)
                self.seqs[seq.name] = seq
            else:
                self.seqs[seq.name] += seq

    def add(self, sequence):
        if sequence.name not in self.keys:
            self.keys.append(sequence.name)
            self.seqs[sequence.name] = sequence
        else:
            self.seqs[sequence.name] += sequence

    def get_sequence(self, k):
        return self.seqs[k]

    def score_intra_sequence(self):
        seqs = [self.seqs[k] for k in self.seqs]
        score = shilouette_score_list_of_group_of_nodes(seqs)
        print("[INFO] Intra Sequence Scroe: {}".format(score))
        return score
    
    def score_inter_sequence(self):
        scores = []
        for k in self.keys:
            seq = self.get_sequence(k)
            score = seq.score_representation()
            scores.append(score)
        score = np.mean(scores)
        print("[INFO] Inter Sequence Score: {}".format(score))
        return score
