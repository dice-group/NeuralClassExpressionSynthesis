import os, random
from utils.syntax_checker import SyntaxChecker
from utils.evaluator import Evaluator
from ontolearn.knowledge_base import KnowledgeBase
from nces import BaseConceptSynthesis
from nces.synthesizer import ConceptSynthesizer
from utils.data import Data
from owlapy.parser import DLSyntaxParser
from dataloader import CSDataLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import torch, pandas as pd
import numpy as np, time
from collections import defaultdict

def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp


def compute_accuracy(prediction, target):
    def soft(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = set(before_pad(BaseConceptSynthesis.decompose(arg1_)))
        else:
            arg1_ = set(before_pad(arg1_))
        if isinstance(arg2_, str):
            arg2_ = set(before_pad(BaseConceptSynthesis.decompose(arg2_)))
        else:
            arg2_ = set(before_pad(arg2_))
        return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

    def hard(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = before_pad(BaseConceptSynthesis.decompose(arg1_))
        else:
            arg1_ = before_pad(arg1_)
        if isinstance(arg2_, str):
            arg2_ = before_pad(BaseConceptSynthesis.decompose(arg2_))
        else:
            arg2_ = before_pad(arg2_)
        return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
    soft_acc = sum(map(soft, prediction, target))/len(target)
    hard_acc = sum(map(hard, prediction, target))/len(target)
    return soft_acc, hard_acc

def map_to_token(model, idx_array):
    return model.inv_vocab[idx_array]

def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    target_tokens_list = []
    target_labels = []
    for pos_emb, neg_emb, label in batch:
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
        target_labels.append(label)
    pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
    return pos_emb_list, neg_emb_list, target_labels

def get_data(kb, embeddings, kwargs):
    data_test_path = f"datasets/{kb}/Test_data/Data.json"
    with open(data_test_path, "r") as file:
        data_test = json.load(file)
    data_test = list(data_test.items())
    test_dataset = CSDataLoader(data_test, embeddings, kwargs)
    print("Number of learning problems: ", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=kwargs.batch_size, num_workers=kwargs.num_workers, collate_fn=collate_batch, shuffle=False)
    return test_dataloader

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
    return prediction

def predict_class_expressions(model_name, kb, args, ensemble=False):
    print(f"\n##{model_name}##")
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    embeddings = pd.read_csv(f"embeddings/{kb}/ConEx_entity_embeddings.csv").set_index('Unnamed: 0')
    dataloader = get_data(kb, embeddings, args)
    if ensemble:
        models = [torch.load(f"datasets/{kb}/Model_weights/{name}.pt", map_location=torch.device('cpu'))\
                  for name in ["SetTransformer", "GRU", "LSTM"]]
    if not ensemble:
        model = torch.load(f"datasets/{kb}/Model_weights/{model_name}.pt", map_location=torch.device('cpu'))
        model.eval()
    else:
        model = models[0]
        model.eval()
    soft_acc, hard_acc = 0.0, 0.0
    preds = []
    targets = []
    for x1, x2, labels in tqdm(dataloader):
        target_sequence = map_to_token(model, labels)
        if ensemble:
            pred_sequence = get_ensemble_prediction(models, x1, x2)
        else:
            pred_sequence, _ = model(x1, x2)
        preds.append(pred_sequence)
        targets.append(target_sequence)
        s_acc, h_acc = compute_accuracy(pred_sequence, target_sequence)
        soft_acc += s_acc
        hard_acc += h_acc
    print(f"Average syntactic accuracy, Soft: {soft_acc/len(dataloader)}%, Hard: {hard_acc/len(dataloader)}%")
    return np.concatenate(preds, 0), np.concatenate(targets, 0)


def evaluate_nces(kb_name, models, args, save_results=False, verbose=False):
    print('#'*50)
    print('NCES evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    desc = ""
    if args.shuffle_examples:
        desc = "_shuffle"
    All_metrics = {m: defaultdict(lambda: defaultdict(list)) for m in models}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'family-benchmark':
        namespace = 'http://www.benchmark.org/family#'
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    syntax_checker = SyntaxChecker(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    All_individuals = set(kb.individuals())
    with open(f"datasets/{kb_name}/Test_data/Data.json", "r") as file:
        data_test = json.load(file)
    for model_name in models:
        t0 = time.time()
        predictions, targets = predict_class_expressions(model_name, kb_name, args)
        t1 = time.time()
        duration = (t1-t0)/len(predictions)
        for i, pb_str in enumerate(targets):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = 1
            pred = predictions[i][:end_idx]
            #print("Before parsing: ", pred.sum())
            succeed = False
            if (pred=='(').sum() > (pred==')').sum():
                for i in range(len(pred))[::-1]:
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,')')))
                        syntax_checker.renderer.render(prediction)
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(pred.sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        print(f"Could not understand expression {pred}")
                        continue
            elif (pred==')').sum() > (pred=='(').sum():
                for i in range(len(pred)):
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,'(')))
                        syntax_checker.renderer.render(prediction)
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(pred.sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        print(f"Could not understand expression {pred}")
                        continue
            else:
                try:
                    prediction = dl_parser.parse_expression("".join(pred.tolist()))
                    syntax_checker.renderer.render(prediction)
                except Exception:
                    try:
                        pred = syntax_checker.correct(pred.sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        print(f"Could not understand expression {pred}")
                        continue
            target_expression = dl_parser.parse_expression(pb_str) # The target class expression
            #positive_examples = {ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(target_expression)}
            #negative_examples = All_individuals-positive_examples
            #acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            try:
                positive_examples = {ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(target_expression)}
                negative_examples = All_individuals-positive_examples
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            except Exception as err:
                print(err)
                continue
            if verbose:
                print(f'Problem {i}, Target: {pb_str}, Prediction: {syntax_checker.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
                print()
            All_metrics[model_name]['acc']['values'].append(acc)
            All_metrics[model_name]['prediction']['values'].append(syntax_checker.renderer.render(prediction))
            All_metrics[model_name]['f1']['values'].append(f1)
            All_metrics[model_name]['time']['values'].append(duration)
            
        for metric in All_metrics[model_name]:
            if metric != 'prediction':
                All_metrics[model_name][metric]['mean'] = [np.mean(All_metrics[model_name][metric]['values'])]
                All_metrics[model_name][metric]['std'] = [np.std(All_metrics[model_name][metric]['values'])]
        
        print(model_name+' Speed: {}s +- {} / lp'.format(round(All_metrics[model_name]['time']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['time']['std'][0], 2)))
        print(model_name+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics[model_name]['acc']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['acc']['std'][0], 2)))
        print(model_name+' Avg F1: {}% +- {} / lp'.format(round(All_metrics[model_name]['f1']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['f1']['std'][0], 2)))
        print()
        if save_results:
            with open("datasets/"+kb_name+"/Results/NCES"+desc+".json", "w") as file:
                json.dump(All_metrics, file, indent=3, ensure_ascii=False)

                
def evaluate_ensemble(kb_name, args, model_name="Ensemble", save_results=False, verbose=False):
    print('#'*50)
    print('NCES evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    All_metrics = {model_name: defaultdict(lambda: defaultdict(list))}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'family-benchmark':
        namespace = 'http://www.benchmark.org/family#'
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    syntax_checker = SyntaxChecker(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    All_individuals = set(kb.individuals())
    with open(f"datasets/{kb_name}/Test_data/Data.json", "r") as file:
        data_test = json.load(file)
        
    t0 = time.time()
    predictions, targets = predict_class_expressions(model_name, kb_name, args, ensemble=True)
    t1 = time.time()
    duration = (t1-t0)/len(predictions)
    for i, pb_str in enumerate(targets):
        pb_str = "".join(before_pad(pb_str))
        try:
            end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
        except IndexError:
            end_idx = 1
        pred = predictions[i][:end_idx]
        #print("Before parsing: ", pred.sum())
        succeed = False
        if (pred=='(').sum() > (pred==')').sum():
            for i in range(len(pred))[::-1]:
                try:
                    prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,')')))
                    succeed = True
                    break
                except Exception:
                    pass
            if not succeed:
                try:
                    pred = syntax_checker.correct(pred.sum())
                    pred = list(syntax_checker.get_suggestions(pred))[-1]
                    prediction = syntax_checker.get_concept(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
                    continue
        elif (pred==')').sum() > (pred=='(').sum():
            for i in range(len(pred)):
                try:
                    prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,'(')))
                    succeed = True
                    break
                except Exception:
                    pass
            if not succeed:
                try:
                    pred = syntax_checker.correct(pred.sum())
                    pred = list(syntax_checker.get_suggestions(pred))[-1]
                    prediction = syntax_checker.get_concept(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
                    continue
        else:
            try:
                prediction = dl_parser.parse_expression("".join(pred.tolist()))
            except Exception:
                try:
                    pred = syntax_checker.correct(pred.sum())
                    pred = list(syntax_checker.get_suggestions(pred))[-1]
                    prediction = syntax_checker.get_concept(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
                    continue
        target_expression = dl_parser.parse_expression(pb_str) # The target class expression
        try:
            positive_examples = {ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(target_expression)}
            negative_examples = All_individuals-positive_examples
            acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
        except Exception as err:
            print(err)
            continue
        if verbose:
            print(f'Problem {i}, Target: {pb_str}, Prediction: {syntax_checker.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
            print()
        All_metrics["Ensemble"]['acc']['values'].append(acc)
        All_metrics["Ensemble"]['prediction']['values'].append(syntax_checker.renderer.render(prediction))
        All_metrics["Ensemble"]['f1']['values'].append(f1)
        All_metrics["Ensemble"]['time']['values'].append(duration)

    for metric in All_metrics["Ensemble"]:
        if metric != 'prediction':
            All_metrics["Ensemble"][metric]['mean'] = [np.mean(All_metrics["Ensemble"][metric]['values'])]
            All_metrics["Ensemble"][metric]['std'] = [np.std(All_metrics["Ensemble"][metric]['values'])]

    print("Ensemble"+' Speed: {}s +- {} / lp'.format(round(All_metrics["Ensemble"]['time']['mean'][0], 2),\
                                                           round(All_metrics["Ensemble"]['time']['std'][0], 2)))
    print("Ensemble"+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics["Ensemble"]['acc']['mean'][0], 2),\
                                                           round(All_metrics["Ensemble"]['acc']['std'][0], 2)))
    print("Ensemble"+' Avg F1: {}% +- {} / lp'.format(round(All_metrics["Ensemble"]['f1']['mean'][0], 2),\
                                                           round(All_metrics["Ensemble"]['f1']['std'][0], 2)))
    
    print()

    if save_results:
        with open("datasets/"+kb_name+"/Results/NCES_Ensemble.json", "w") as file:
            json.dump(All_metrics, file, indent=3, ensure_ascii=False)
