import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('utils')[0])
from nces import BaseConceptSynthesis
import numpy as np, torch, pandas as pd
from .data import Data

class CSDataLoader(BaseConceptSynthesis, Data, torch.utils.data.Dataset):
    def __init__(self, data, embeddings, kwargs):
        self.data_raw = data
        self.embeddings = embeddings
        super().__init__(kwargs)
        self.vocab_df = pd.DataFrame(self.vocab.values(), index=self.vocab.keys())

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = sorted(value['positive examples'])
        neg = sorted(value['negative examples'])
        assert '#' in pos[0] or '.' in pos[0], 'Namespace error, expected separator # or .'
        datapoint_pos = torch.FloatTensor(self.embeddings.loc[pos].values)
        datapoint_neg = torch.FloatTensor(self.embeddings.loc[neg].values)
        labels, length = self.get_labels(key)
        #torch.cat([datapoint_pos, datapoint_neg], 0)
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(self.max_length-length)]).long()
            
        