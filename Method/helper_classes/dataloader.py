import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0])
from base import BaseConceptSynthesis
from functools import singledispatchmethod
import numpy as np, random, torch, pandas as pd
from typing import List
from util.data import Data
from abc import ABCMeta
random.seed(1)

class CSDataLoader(BaseConceptSynthesis, Data, torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(self, data, embeddings, kwargs):
        self.data_raw = data
        self.embeddings = embeddings
        super().__init__(kwargs)

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        assert '#' in pos[0] or '.' in pos[0], 'Namespace error, expected separator # or .'
        datapoint_pos = torch.FloatTensor(list(map(lambda x: self.embeddings.loc[x], pos))).mean(0).unsqueeze(0)
        datapoint_neg = torch.FloatTensor(list(map(lambda x: self.embeddings.loc[x], neg))).mean(0).unsqueeze(0)
        target, numerical_target = self.get_scores_of_atom_indices(key)
        return torch.cat([datapoint_pos, datapoint_neg], 0), numerical_target, \
    torch.cat([torch.tensor(list(map(lambda x: self.vocab[x], target))), -100*torch.ones(self.max_num_atom_repeat-len(target))], 0).long()
            
            
class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec
            
