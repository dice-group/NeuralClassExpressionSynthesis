import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import sys, os, json
import argparse

base_path = os.path.dirname(os.path.realpath(__file__)).split('reproduce_results')[0]
sys.path.append(base_path)

from helper_classes.experiment import Experiment
from util.data import Data
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Ivalid boolean value.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--kb', type=str, default='carcinogenesis', help='Knowledge base name')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use to load training data')
parser.add_argument('--rnn_n_hidden', type=int, default=100, help='Hidden size of recurrent neural networks')
parser.add_argument('--num_examples', type=int, default=1000, help='Total number of examples for concept learning')
parser.add_argument('--decay_rate', type=float, default=1e-5, help='Decay rate for the optimizer')
parser.add_argument('--grad_clip_value', type=float, default=5.0, help='Gradient clip value')
parser.add_argument('--optimizer', type=str, default='Adam', help='Name of the optimizer to use')
parser.add_argument('--model', type=str, default='GRU', help='Name of the concept learner architecture to use')
parser.add_argument('--max_num_atom_repeat', type=int, default=64, help='Maximum number of atom repetition in a given class expression')
parser.add_argument('--rnn_n_layers', type=int, default=2, help='Number of recurrent network layers')
parser.add_argument('--index_score_upper_bound', type=float, default=10.0, help='Upper bound for scoring atoms/tokens')
parser.add_argument('--index_score_lower_bound_rate', type=float, default=0.8, help='Lower bound rate')
parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout rate in neural networks')

args = parser.parse_args()


data_train_path = base_path+f"Datasets/{args.kb}/Train_data/Data.json"
with open(data_train_path, "r") as file:
    data_train = json.load(file)
    
data_test_path = base_path+f"Datasets/{args.kb}/Test_data/Data.json"
with open(data_test_path, "r") as file:
    data_test = json.load(file)
path_to_triples = base_path+f"Datasets/{args.kb}/Triples/"
triples = Data({"path_to_triples": path_to_triples})

kwargs = {"learner_name": args.model, "emb_model_name":"", 'knowledge_base_path': base_path+f"Datasets/{args.kb}/{args.kb}.owl",
          "pretrained_embedding_path": base_path+f"Datasets/{args.kb}/Model_weights/ConEx_GRU.pt",
          "pretrained_concept_synthesizer": base_path+f"Datasets/{args.kb}/Model_weights/GRU.pt", 
          "path_to_csv_embeddings": base_path+f"Embeddings/{args.kb}/ConEx_entity_embeddings.csv",
          "learning_rate": args.lr, "decay_rate": args.decay_rate, 'grad_clip_value': args.grad_clip_value, 
          "path_to_triples": path_to_triples, 'max_num_atom_repeat': args.max_num_atom_repeat,
          'index_score_upper_bound': args.index_score_upper_bound, 'index_score_lower_bound_rate': args.index_score_lower_bound_rate,
          "embedding_dim": 20, "num_entities": len(triples.entities),
          "num_relations": len(triples.relations), "num_examples": args.num_examples, 'drop_prob': args.drop_prob,
          "rnn_n_layers": args.rnn_n_layers, 'input_size': 40, 'rnn_n_hidden': args.rnn_n_hidden}

Models = ["GRU", "LSTM"]

experiment = Experiment(kwargs)

data_train, data_test = list(data_train.items()), list(data_test.items())

final = False
test = True
cross_validate = True
record_runtime = True
save_model = True
if final:
    data_train = data_train + data_test
    test = False
    cross_validate = False
experiment.train_all_nets(Models, data_train, data_test, epochs=500, batch_size=128, kf_n_splits=10, 
                          cross_validate=cross_validate, test=test, save_model = save_model,
                          optimizer = 'Adam', record_runtime=record_runtime, final=final)
