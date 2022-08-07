import torch, torch.nn as nn, numpy as np

class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        self.Phi1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], kwargs['proj_dim'])),
                         dtype=torch.float, requires_grad=True))
        self.Phi2 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], kwargs['proj_dim'])),
                         dtype=torch.float, requires_grad=True))
        # nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.lstm1 = nn.LSTM(kwargs['proj_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.lstm2 = nn.LSTM(kwargs['proj_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 5*kwargs['rnn_n_hidden']), nn.BatchNorm1d(5*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(5*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x1, x2, target_scores=None):
        x1 = x1.matmul(self.Phi1)
        x2 = x2.matmul(self.Phi2)
        seq1, _ = self.lstm1(x1)
        seq2, _ = self.lstm2(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out2 = seq2.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        x = out1*out2
        x = self.fc(x).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
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
                num_select = max(1,(x[i]>=min(target_scores[i][target_scores[i]!=0.])).sum().item())
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
        
        self.Phi1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], kwargs['proj_dim'])),
                         dtype=torch.float, requires_grad=True))
        self.Phi2 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], kwargs['proj_dim'])),
                         dtype=torch.float, requires_grad=True))
        self.gru1 = nn.GRU(kwargs['proj_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.gru2 = nn.GRU(kwargs['proj_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 5*kwargs['rnn_n_hidden']), nn.BatchNorm1d(5*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(5*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x1, x2, target_scores=None):
        x1 = x1.matmul(self.Phi1)
        x2 = x2.matmul(self.Phi2)
        seq1, _ = self.gru1(x1)
        seq2, _ = self.gru2(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out2 = seq2.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        x = out1*out2
        x = self.fc(x).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
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
                num_select = max(1,(x[i]>=min(target_scores[i][target_scores[i]!=0.])).sum().item())
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
        super().__init__()
        self.kwargs = kwargs
        self.name = 'DeepSet'
        
        self.Phi1 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], 768)),
                         dtype=torch.float, requires_grad=True))
        self.Phi2 = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (kwargs['input_size'], 768)),
                         dtype=torch.float, requires_grad=True))
        self.fc = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                nn.Linear(1024, kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
        #self.relu = nn.ReLU()
    
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