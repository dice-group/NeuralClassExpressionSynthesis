"""
Deploying neural class expression synthesizers (NCES)

"""
from typing import Union, List

import pandas as pd
import torch
import json
import gradio as gr
from argparse import ArgumentParser, Namespace
import random
from ontolearn.knowledge_base import KnowledgeBase
from utils.dataloader import NCESDataLoaderDeploy
from torch.utils.data import DataLoader
from owlapy.parser import DLSyntaxParser
from owlapy.render import DLSyntaxObjectRenderer
from utils.evaluator import Evaluator
from utils.simple_solution import SimpleSolution
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def get_data(kb, embeddings, test_data, num_examples, kwargs, selected_lps):
    test_data_ = [test_data[i] for i in selected_lps]
    test_dataset = NCESDataLoaderDeploy(test_data_, embeddings, kwargs)
    def collate_batch(batch):
        pos_emb_list = []
        neg_emb_list = []
        for pos_emb, neg_emb in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        return pos_emb_list, neg_emb_list
    test_dataloader = DataLoader(test_dataset, batch_size=kwargs.batch_size, num_workers=kwargs.num_workers, collate_fn=collate_batch, shuffle=False)
    return test_data_, test_dataloader

def get_ensemble_prediction(models, x1, x2):
    for i,model in enumerate(models):
        model.eval()
        if i == 0:
            _, scores = model(x1, x2)
        else:
            _, sc = model(x1, x2)
            scores = scores + sc
    scores = scores/len(models)
    prediction = model.inv_vocab[scores.argmax(1)]
    return prediction, scores

def predict_single(model, pos_emb, neg_emb):
    return model(pos_emb, neg_emb)


def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return "".join(arg_temp)

def launch_service(nces_model: Union[List,str], kb, kwargs):
    if isinstance(nces_model, str):
        model = torch.load(f"datasets/{kb}/Model_weights/ConEx_{model_name}.pt", map_location=torch.device('cpu'))
    else:
        models = [torch.load(f"datasets/{kb}/Model_weights/ConEx_{name}.pt", map_location=torch.device('cpu'))\
              for name in nces_model]
    embeddings = pd.read_csv(f"datasets/{kb}/Embeddings/ConEx_entity_embeddings.csv").set_index('Unnamed: 0')
    test_data_path = f"datasets/{kb}/Test_data/Data.json"
    with open(test_data_path, "r") as file:
        test_data = list(json.load(file).items())
    max_num_lps = len(test_data)
    KB = KnowledgeBase(path=f"datasets/{kb}/{kb}.owl")
    
    renderer = DLSyntaxObjectRenderer()
    num_examples = min(KB.individuals_count()//2, 1000)
    evaluator = Evaluator(KB)
    simpleSolution = SimpleSolution(KB)
    all_individuals = set(KB.individuals())
    kb_namespace = list(KB.individuals())[0].get_iri().get_namespace()
    dl_parser = DLSyntaxParser(namespace = kb_namespace)
    def predict(lp_ids, number_of_learning_problems: int, random_problems: bool):
        kwargs.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
        number_of_learning_problems = int(number_of_learning_problems)
        if random_problems:
            input_ids = sorted(random.sample(list(range(max_num_lps)), number_of_learning_problems))
            dataset, dataloader = get_data(kb, embeddings, test_data, num_examples, kwargs, input_ids)
        else:
            input_ids = sorted(list({int(idx) for idx in lp_ids.split(",") if int(idx)<max_num_lps}))
            dataset, dataloader = get_data(kb, embeddings, test_data, num_examples, kwargs, input_ids)
        pos_emb, neg_emb = next(iter(dataloader))
        if isinstance(nces_model, str):
            predictions, _ = predict_single(model, pos_emb, neg_emb)
        else:
            predictions, _ = get_ensemble_prediction(models, pos_emb, neg_emb)
        results = list(map(lambda pred: before_pad(pred), predictions))
        print("\nNumber of learning problems: ", len(input_ids), "\n")
        ## Evaluate solutions
        out_text = ""
        output_results = {"IDs": input_ids, "Prediction": [], "Acc": [], "F1": []}
        for i in range(len(results)):
            target_expr, examples = dataset[i]
            prediction_str = results[i]
            try:
                prediction = dl_parser.parse_expression(prediction_str)
            except Exception:
                pred = simpleSolution.predict(prediction_str)
                prediction = dl_parser.parse_expression(pred)
                prediction_str = renderer.render(prediction)
            target_expression = dl_parser.parse_expression(target_expr) # The target class expression
            positive_examples = set(KB.individuals(target_expression))
            negative_examples = all_individuals-positive_examples
            acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            output_results["Prediction"].append(prediction_str)
            output_results["Acc"].append(acc)
            output_results["F1"].append(f1)
            pos = list(map(lambda x: kb_namespace+x.split('#')[-1],examples['positive examples'][:5]))
            neg = list(map(lambda x: kb_namespace+x.split('#')[-1], examples['negative examples'][:5]))
            out_text += f"ID: {input_ids[i]}\n" + f"|E^+|={len(examples['positive examples'])}: {'['+','.join(pos)+',...]'}\n|E^-|={len(examples['negative examples'])}: {'['+','.join(neg)+',...]'}\n\n"
            
        return out_text, pd.DataFrame(output_results)
    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label=f'Learning Problem IDs on Test Set for {kb.upper()} KB (ID<{max_num_lps})'),
                gr.inputs.Slider(minimum=1, maximum=max_num_lps),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'), gr.outputs.Dataframe(label='Predictions')],
        title='Neural Class Expression Synthesis (NCES)',
        description='Select random_problems and submit or enter learning problem IDs (separated with a comma if many)').launch(share=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')

if __name__ == '__main__':
    with open("settings.json") as setting:
        args = json.load(setting)
    args = Namespace(**args)
    
    parser = ArgumentParser()
    parser.add_argument("--ensemble", type=str2bool, default=True,
                        help='Whether to use model ensembling')
    parser.add_argument("--model", type=str, default="SetTransformer",
                        help='Name of the synthesizer')
    parser.add_argument("--kb", type=str, default="carcinogenesis",
                        help='Name of the knowledge base')
    
    Args = parser.parse_args()
    kb = Args.kb
    if Args.ensemble:
        launch_service(["SetTransformer", "GRU", "LSTM"], kb, args)
    else:
        launch_service(Args.model, kb, args)
