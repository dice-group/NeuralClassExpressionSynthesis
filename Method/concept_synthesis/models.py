import torch, torch.nn as nn, numpy as np

class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        
        self.lstm = nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x, target_scores=None):
        r_out, _ = self.lstm(x)
        out = r_out.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>=min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out

        
class ConceptLearner_GRU(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU'
        
        self.gru = nn.GRU(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x, target_scores=None):
        r_out, _ = self.gru(x)
        out = r_out.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>=min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out

class ConceptLearner_CNN(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'CNN'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(kwargs['kernel_h'],kwargs['kernel_w']), stride=(kwargs['stride_h'],kwargs['stride_w']))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(kwargs['kernel_h']+1,kwargs['kernel_w']), stride=(kwargs['stride_h']+2,kwargs['stride_w']+1))
        self.dropout2d = nn.Dropout2d(kwargs['drop_prob'])
        self.fc = nn.Sequential(nn.Linear(kwargs['conv_out'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())  
    def forward(self, x, target_scores=None):
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.dropout2d(out)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        #print("shape", x.shape)
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>=min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out
    
    
    
class DeepSet(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'DeepSet'
        
        self.Phi1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], 256)),
                         dtype=torch.float, requires_grad=True))
        self.Phi2 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (256, 512)),
                         dtype=torch.float, requires_grad=True))
        self.fc = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024),
                                nn.Linear(1024, kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
        #self.relu = nn.ReLU()
    
    def forward(self, x, target_scores=None):
        x = x.matmul(self.Phi1)
        x = x.matmul(self.Phi2)
        x = x.mean(1).view(-1, x.shape[2])
        #x = self.relu(x)
        x = self.fc(x).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>0.9*self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
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