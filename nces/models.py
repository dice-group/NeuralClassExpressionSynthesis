import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from .modules import *
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('concept_synthesis')[0]
sys.path.append(base_path)
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import pandas as pd

class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        vocab = sorted(vocab) + ['PAD']
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(kwargs.input_size, kwargs.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs.proj_dim)
        self.fc1 = nn.Linear(2*kwargs.proj_dim, kwargs.proj_dim)
        self.fc2 = nn.Linear(kwargs.proj_dim, kwargs.proj_dim)
        self.fc3 = nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length)
        
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.lstm(x1)
        seq2, _ = self.lstm(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs.proj_dim)
        out2 = seq2.sum(1).view(-1, self.kwargs.proj_dim)
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = F.relu(self.fc3(x))
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1)]
        return aligned_chars, x

        
class ConceptLearner_GRU(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU'
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        vocab = sorted(vocab) + ['PAD']
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.gru = nn.GRU(kwargs.input_size, kwargs.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs.proj_dim)
        self.fc1 = nn.Linear(2*kwargs.proj_dim, kwargs.proj_dim)
        self.fc2 = nn.Linear(kwargs.proj_dim, kwargs.proj_dim)
        self.fc3 = nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length)
    
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.gru(x1)
        seq2, _ = self.gru(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs.proj_dim)
        out2 = seq2.sum(1).view(-1, self.kwargs.proj_dim)
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = F.relu(self.fc3(x))
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1)]
        return aligned_chars, x

    
    
class SetTransformer(nn.Module):
    def __init__(self, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.kwargs = kwargs
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        vocab = sorted(vocab) + ['PAD']
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.enc = nn.Sequential(
                ISAB(kwargs.input_size, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln),
                ISAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln))
        self.dec = nn.Sequential(
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                SAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length))

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1)]
        return aligned_chars, x

    
class TreeTransformer(nn.Module):
    def __init__(self, kwargs):
        super(TreeTransformer, self).__init__()
        self.name = 'TreeTransformer'
        self.kwargs = kwargs
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥']
        vocab = sorted(vocab)
        self.inv_vocab = vocab
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        
        self.embedding = nn.Embedding(len(self.vocab), kwargs.embedding_dim)
        self.encoder = nn.Sequential(
                ISAB(kwargs.input_size, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln),
                ISAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln))
        self.decoder_head = nn.Sequential(
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                SAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)))
        self.decoder_unary = nn.Sequential(
                MAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln),
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)))
        self.decoder_left = nn.Sequential(
                MAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln),
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)))
        self.dec_right = nn.Sequential(
                MAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln),
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)))
        
    def forward(self, x1, x2, target_scores=None):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.kwargs.max_num_atom_repeat)
        values, sorted_indices = x.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(x[i]>=0.9*self.kwargs.alpha*(1-self.kwargs.lbr)).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.vocab[(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.vocab[(x[i]==v).nonzero().squeeze()[0][0].item()])
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
                        atoms.append(self.vocab[(x[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.vocab[(x[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, x