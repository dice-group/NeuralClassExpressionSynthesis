import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from .modules import *

class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        # nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.lstm = nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs['proj_dim'])
        self.fc1 = nn.Linear(2*kwargs['rnn_n_hidden'], kwargs['proj_dim'])
        self.fc2 = nn.Linear(kwargs['proj_dim'], kwargs['proj_dim'])
        self.fc3 = nn.Linear(kwargs['proj_dim'], kwargs['output_size']*kwargs['max_num_atom_repeat'])
        
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.lstm(x1)
        seq2, _ = self.lstm(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out2 = seq2.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = F.relu(self.fc3(x))
        x = x.reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        #self.fc(x).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.8*self.kwargs['alpha']*(1-self.kwargs['lbr'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>=0.8*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x

        
class ConceptLearner_GRU(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU'
        self.gru = nn.GRU(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs['proj_dim'])
        self.fc1 = nn.Linear(2*kwargs['rnn_n_hidden'], kwargs['proj_dim'])
        self.fc2 = nn.Linear(kwargs['proj_dim'], kwargs['proj_dim'])
        self.fc3 = nn.Linear(kwargs['proj_dim'], kwargs['output_size']*kwargs['max_num_atom_repeat'])
    
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.gru(x1)
        seq2, _ = self.gru(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out2 = seq2.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = F.relu(self.fc3(x))
        x = x.reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.8*self.kwargs['alpha']*(1-self.kwargs['lbr'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>=0.8*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x

    
    
class DeepSet0(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'DeepSet0'
        
        self.Phi1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], 768)),
                         dtype=torch.float, requires_grad=True))
        self.Phi2 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], 768)),
                         dtype=torch.float, requires_grad=True))
        self.fc = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                nn.Linear(1024, kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    def forward(self, x1, x2, target_scores=None):
        x1 = x1.matmul(self.Phi1)
        x2 = x2.matmul(self.Phi2)
        x1 = x1.sum(1).view(-1, x1.shape[2]) # shape (batch_size, 1024)
        x2 = x2.sum(1).view(-1, x2.shape[2]) # shape (batch_size, 1024)
        x = x1*x2
        #x = torch.cat([x1*x2, x1+x2, x1-x2, x1/x2, x1, x2], dim=1)
        x = self.fc(x).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*self.kwargs['alpha']*(1-self.kwargs['lbr'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x
    

class DeepSet(nn.Module):
    def __init__(self, kwargs):
        super(DeepSet, self).__init__()
        self.kwargs = kwargs
        self.name = 'DeepSet'
        self.enc = nn.Sequential(
                nn.Linear(kwargs['input_size'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['proj_dim']))
        self.dec = nn.Sequential(
                nn.Linear(2*kwargs['proj_dim'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['proj_dim']),
                nn.ReLU(),
                nn.Linear(kwargs['proj_dim'], kwargs['output_size']*kwargs['max_num_atom_repeat']))

    def forward(self, x1, x2, target_scores=None):
        x1 = self.enc(x1).mean(-2)
        x2 = self.enc(x2).mean(-2)
        x = torch.cat([x1,x2], -1)
        x = self.dec(x).reshape(-1, len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*self.kwargs['alpha']*(1-self.kwargs['lbr'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x

    
class SetTransformer(nn.Module):
    def __init__(self, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.kwargs = kwargs
        self.enc = nn.Sequential(
                ISAB(kwargs['input_size'], kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_inds'], ln=kwargs['ln']),
                ISAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_inds'], ln=kwargs['ln']))
        self.dec = nn.Sequential(
                PMA(kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_seeds'], ln=kwargs['ln']),
                SAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], ln=kwargs['ln']),
                SAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], ln=kwargs['ln']),
                nn.Linear(kwargs['proj_dim'], kwargs['output_size']*kwargs['max_num_atom_repeat']))

    def forward(self, x1, x2, target_scores=None):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>=0.9*self.kwargs['alpha']*(1-self.kwargs['lbr'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x

    
class TreeTransformer(nn.Module):
    def __init__(self, kwargs):
        super(TreeTransformer, self).__init__()
        self.name = 'TreeTransformer'
        
        self.kwargs = kwargs
        self.enc = nn.Sequential(
                ISAB(kwargs['input_size'], kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_inds'], ln=kwargs['ln']),
                ISAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_inds'], ln=kwargs['ln']))
        self.dec_head = nn.Sequential(
                PMA(kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_seeds'], ln=kwargs['ln']),
                SAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], ln=kwargs['ln']),
                nn.Linear(kwargs['proj_dim'], kwargs['output_size']))
        self.dec_left = nn.Sequential(
                PMA(kwargs['proj_dim'], kwargs['num_heads'], kwargs['num_seeds'], ln=kwargs['ln']),
                SAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], ln=kwargs['ln']),
                SAB(kwargs['proj_dim'], kwargs['proj_dim'], kwargs['num_heads'], ln=kwargs['ln']),
                nn.Linear(kwargs['proj_dim'], kwargs['output_size']))