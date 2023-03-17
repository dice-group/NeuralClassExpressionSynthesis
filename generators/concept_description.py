import random, numpy as np
from tqdm import tqdm

class ConceptDescriptionGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, depth=2, max_length=10, num_rand_samples=150):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.depth = depth
        self.num_rand_samples = num_rand_samples
        self.max_length = max_length

    def apply_rho(self, concept):
        return {ref for ref in self.rho.refine(concept, max_length=self.max_length)}

    def generate(self):
        roots = self.apply_rho(self.kb.thing)
        ## Now remove redundant expressions before proceeding!
        Refinements = set()
        Refinements.update(roots)
        print ("|Thing refinements|: ", len(roots))
        roots_sample = random.sample(list(roots), k=self.num_rand_samples)
        print("Number of roots: ", len(roots_sample))
        for root in tqdm(roots_sample, desc="Refining roots..."):
            Refinements.update(self.apply_rho(root))
        return Refinements
