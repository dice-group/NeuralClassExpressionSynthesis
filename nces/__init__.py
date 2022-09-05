import os, torch, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('base')[0])
import matplotlib.pyplot as plt
from ontolearn.knowledge_base import KnowledgeBase
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer

class BaseConceptSynthesis:
    """Supervised Machine Learning approach for learning class expressions in ALC from examples"""
    
    def __init__(self, kwargs):
        self.knowledge_base_path = kwargs.knowledge_base_path
        kb = KnowledgeBase(path=self.knowledge_base_path)
        self.__num_examples__ = min(kwargs.num_examples, kb.individuals_count()//2)
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.alpha = kwargs.alpha
        self.lbr = kwargs.lbr
        atomic_concepts = list(kb.ontology().classes_in_signature())
        self.atomic_concept_names = [self.dl_syntax_renderer.render(a) for a in atomic_concepts]
        self.role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()]
        vocab = self.atomic_concept_names + self.role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')']
        vocab = sorted(vocab)
        self.inv_vocab = vocab
        self.vocab = {vocab[i]:i for i in range(len(vocab))} #dict(map(reversed, enumerate(vocab)))
        self.max_num_atom_repeat = kwargs.max_num_atom_repeat
        self.max_length = kwargs.max_length
        self.kwargs = kwargs
        
    @property
    def num_examples(self):
        return self.__num_examples__
    
    @staticmethod
    def decompose(concept_name: str) -> list:
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in ['(', ')', '⊔', '⊓', '∃', '∀', '¬', '.', ' ']:
                concept += concept_name[i]
                i += 1
            if concept and i < len(concept_name):
                list_ordered_pieces.extend([concept, concept_name[i]])
            elif concept:
                list_ordered_pieces.append(concept)
            elif i < len(concept_name):
                list_ordered_pieces.append(concept_name[i])
            i += 1
        return list_ordered_pieces
    
    def get_scores_of_atom_indices(self, target):
        Scores = torch.zeros((len(self.vocab),self.max_num_atom_repeat))
        target = self.decompose(target)
        if self.kwargs.use_adaptive_bounds:
            scores = torch.tensor(len(target)).sqrt()*torch.linspace(self.alpha, self.alpha*(1-self.lbr), len(target))
        else:
            scores = torch.linspace(self.alpha, self.alpha*(1-self.lbr), len(target))
        atom_counts = {a: 0 for a in target}
        for j in range(len(target)):
            try:
                Scores[self.vocab[target[j]], atom_counts[target[j]]] = scores[j]
                atom_counts[target[j]] += 1
            except IndexError:
                print('Index out of bound error, ignoring current atom index')
        return target, Scores