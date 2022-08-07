import torch, random
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('concept_synthesis')[0]
sys.path.append(base_path)
from concept_synthesis.models import *
from Embeddings.models import *
#from owlapy.model import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from typing import Set, List, Union, Final
from sklearn.utils import resample
import pandas as pd

class ConceptSynthesizer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        kb = KnowledgeBase(path=kwargs['knowledge_base_path'])
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        atomic_concepts: Final = frozenset(kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        self.role_names: Final = frozenset([rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()])
        vocab = list(self.atomic_concept_names) + list(self.role_names) + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        self.inv_vocab = vocab
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.learner_name = kwargs['learner_name'] if kwargs['learner_name'] else "DeepSet"
        self.synthesizer = self.get_synthesizer()
        self.embedding_model = self.get_embedding_model(kwargs['emb_model_name'])
        
    def get_embedding_model(self, name=""):
        if name == 'ConEx':
            return ConEx(self.kwargs)
        elif name == 'Complex':
            return Complex(self.kwargs)
        elif name == 'Distmult':
            return Distmult(self.kwargs)
        elif name == 'Tucker':
            return Tucker(self.kwargs)
        else:
            print('No embedding model given, will require pretrained embeddings in csv format')
    
    def get_embedding(self, embedding_model=None):
        if embedding_model:
            if embedding_model.name == 'ConEx':
                return (embedding_model.emb_ent_real, embedding_model.emb_ent_i)
            elif embedding_model.name == 'Complex':
                return (embedding_model.Er, embedding_model.Ei)
            elif embedding_model.name == 'Distmult':
                return (embedding_model.emb_ent_real,)
            elif embedding_model.name == 'Tucker':
                return (embedding_model.E,)
        return pd.read_csv(self.kwargs['path_to_csv_embeddings']).set_index('Unnamed: 0')
    
    def get_synthesizer(self):
        self.kwargs['vocab'] = list(self.vocab.keys())
        self.kwargs['output_size'] = len(self.kwargs['vocab'])
        if self.learner_name == 'DeepSet':
            return DeepSet(self.kwargs)
        elif self.learner_name == 'GRU':
            return ConceptLearner_GRU(self.kwargs)
        elif self.learner_name == 'LSTM':
            return ConceptLearner_LSTM(self.kwargs)
        elif self.learner_name == 'CNN':
            return ConceptLearner_CNN(self.kwargs)
        else:
            print('Wrong concept learner name')
            raise ValueError
            
    def refresh(self):
        self.synthesizer = self.get_synthesizer()
        
    def load_pretrained(self):
        assert self.kwargs['pretrained_concept_synthesizer'], 'No pretrained length learner'
        self.synthesizer = torch.load(self.kwargs['pretrained_concept_synthesizer'], map_location=torch.device('cpu'))
        self.synthesizer.eval()
        if self.embedding_model:
            assert self.kwargs['pretrained_embedding_model'], 'No pretrained embedding model'
            self.embedding_model = torch.load(self.kwargs['pretrained_embedding_model'], map_location=torch.device('cpu'))
            self.embedding_model.eval()
        
    #def predict(self, pos: Union[List[OWLNamedIndividual], List[str]]):
    #    self.load_pretrained()
    #    if isinstance(pos[0], OWLNamedIndividual):
    #        pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
    #    if len(pos) > self.kwargs['num_examples']:
    #        p = random.sample(pos, k=self.kwargs['num_examples'])
    #    elif len(pos) == self.kwargs['num_examples']:
    #        p = pos
    #    else:
    #        p = resample(pos, replace=True, n_samples=self.kwargs['num_examples'], random_state=123)
    #    datapoint = [(" ", {"positive examples": p})]
    #    if self.embedding_model is not None:
    #        x, _, _ = list(self.dataloader.load(self.get_embedding(self.embedding_model), datapoint, 1, False))[0]
    #    else:
    #        x, _, _ = self.dataloader.load(self.get_embedding(self.embedding_model), datapoint, 1, False)
    #    return self.synthesizer(x)
    #
    
    