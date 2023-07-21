"""
Deploying Evolearner

"""
import pandas as pd
import json
import gradio as gr
from argparse import ArgumentParser
import random
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from utils.evaluator import Evaluator
from owlapy.parser import DLSyntaxParser
from owlapy.render import DLSyntaxObjectRenderer
import warnings
warnings.filterwarnings("ignore")


def launch_service(kb, kwargs):
    with open(f"datasets/{kb}/Test_data/Data.json", "r") as file:
        data_test = list(json.load(file).items())
    max_num_lps = len(data_test)
    KB = KnowledgeBase(path=f'datasets/{kb}/{kb}.owl')
    kb_namespace = list(KB.ontology().classes_in_signature())[0].get_iri().get_namespace()
    kb_prefix = kb_namespace[:kb_namespace.rfind("/")+1]
    evaluator = Evaluator(KB)
    renderer = DLSyntaxObjectRenderer()
    dl_parser = DLSyntaxParser(namespace = kb_namespace)
    all_individuals = set(KB.individuals())
    def predict(lp_ids, number_of_learning_problems: int, random_problems: bool):
        number_of_learning_problems = int(number_of_learning_problems)
        if random_problems:
            input_ids = sorted(random.sample(list(range(max_num_lps)), number_of_learning_problems))
        else:
            input_ids = sorted(list({int(idx) for idx in lp_ids.split(",") if int(idx)<max_num_lps}))
        dataset = [data_test[i] for i in input_ids]
        print("\nNumber of learning problems: ", len(input_ids), "\n")
        out_text = ""
        output_results = {"IDs": input_ids, "Prediction": [], "Acc": [], "F1": []}
        for i, (str_target_concept, examples) in enumerate(dataset):
            pos = list(map(lambda x: kb_namespace+x.split('#')[-1],examples['positive examples'][:5]))
            neg = list(map(lambda x: kb_namespace+x.split('#')[-1], examples['negative examples'][:5]))
            out_text += f"ID: {input_ids[i]}\n" + f"|E^+|={len(examples['positive examples'])}: {'['+','.join(pos)+',...]'}\n|E^-|={len(examples['negative examples'])}: {'['+','.join(neg)+',...]'}\n\n"
            model = EvoLearner(knowledge_base=KB, max_runtime=kwargs.timeout)
            p = [kb_prefix+ind for ind in examples['positive examples']]
            n = [kb_prefix+ind for ind in examples['negative examples']]
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            model.fit(lp)
            for desc in model.best_hypotheses(1):
                target_expression = dl_parser.parse_expression(str_target_concept) # The target class expression
                positive_examples = set(KB.individuals(target_expression))
                negative_examples = all_individuals-positive_examples
                acc, f1 = evaluator.evaluate(desc.concept, positive_examples, negative_examples)
                prediction_str = renderer.render(desc.concept)
                output_results["Prediction"].append(prediction_str)
                output_results["Acc"].append(acc)
                output_results["F1"].append(f1)
        return out_text, pd.DataFrame(output_results)
    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label=f'Learning Problem IDs on Test Set for {kb.upper()} KB (ID<{max_num_lps})'),
                gr.inputs.Slider(minimum=1, maximum=max_num_lps),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'), gr.outputs.Dataframe(label='Predictions')],
        title='Class Expression Learning with Evolearner',
        description='Select random_problems and submit or enter learning problem IDs (separated with a comma if many)').launch(share=True)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--timeout", type=float, default=300,
                        help='The maximum execution time of EvoLearner')
    parser.add_argument("--kb", type=str, default="carcinogenesis",
                        help='Name of the knowledge base')
    args = parser.parse_args()
    launch_service(args.kb, args)