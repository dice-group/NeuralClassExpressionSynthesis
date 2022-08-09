import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import sys, os, json
import argparse
import random

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        raise ValueError('Invalid boolean value.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--kb', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use to load training data')
parser.add_argument('--rnn_n_hidden', type=int, default=256, help='Hidden size of recurrent neural networks')
parser.add_argument('--proj_dim', type=int, default=256, help='The projection dimension for examples')
parser.add_argument('--num_examples', type=int, default=1000, help='Total number of examples for concept learning')
parser.add_argument('--decay_rate', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--grad_clip_value', type=float, default=5.0, help='Gradient clip value')
parser.add_argument('--opt', type=str, default='Adam', help='Name of the optimizer to use')
parser.add_argument('--max_num_atom_repeat', type=int, default=8, help='Maximum number of an atom repetition in a given class expression')
parser.add_argument('--rnn_n_layers', type=int, default=2, help='Number of recurrent network layers')
parser.add_argument('--index_score_upper_bound', type=float, default=10.0, help='Upper bound for scoring atoms/tokens')
parser.add_argument('--index_score_lower_bound_rate', type=float, default=0.8, help='Lower bound rate')
parser.add_argument('--max_length', type=int, default=32, help='Maximum length of class expressions')
parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout rate in neural networks')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--cross_validate', type=str2bool, default=False, help='Whether to use a 10-fold cross-validation setting')
parser.add_argument('--test', type=str2bool, default=True, help='Whether to evaluate the concept synthesizer on the test data during training')
parser.add_argument('--final', type=str2bool, default=False, help='Whether to train the concept synthesizer on test+train data')

args = parser.parse_args()

for kb in args.kb:
    data_train_path = base_path+f"Datasets/{kb}/Train_data/Data.json"
    with open(data_train_path, "r") as file:
        data_train = json.load(file)

    data_test_path = base_path+f"Datasets/{kb}/Test_data/Data.json"
    with open(data_test_path, "r") as file:
        data_test = json.load(file)
    path_to_triples = base_path+f"Datasets/{kb}/Triples/"
    triples = Data({"path_to_triples": path_to_triples})

    kwargs = {"learner_name": "", "emb_model_name": "", 'knowledge_base_path': base_path+f"Datasets/{kb}/{kb}.owl",
              "pretrained_embedding_path": base_path+f"Datasets/{kb}/Model_weights/ConEx_GRU.pt",
              "pretrained_concept_synthesizer": base_path+f"Datasets/{kb}/Model_weights/GRU.pt", 
              "path_to_csv_embeddings": base_path+f"Embeddings/{kb}/ConEx_entity_embeddings.csv",
              "learning_rate": args.lr, "decay_rate": args.decay_rate, 'grad_clip_value': args.grad_clip_value, 
              "path_to_triples": path_to_triples, 'max_num_atom_repeat': args.max_num_atom_repeat,
              'index_score_upper_bound': args.index_score_upper_bound, 'index_score_lower_bound_rate': args.index_score_lower_bound_rate,
              'max_length': args.max_length, 'num_workers': args.num_workers,
              "embedding_dim": 20, "num_entities": len(triples.entities),
              "num_relations": len(triples.relations), "num_examples": args.num_examples, 'drop_prob': args.drop_prob,
              "rnn_n_layers": args.rnn_n_layers, 'input_size': 40, 'rnn_n_hidden': args.rnn_n_hidden,
              "proj_dim": args.proj_dim}

    Models = ["LSTM", "GRU"]

    experiment = Experiment(kwargs)

    data_train, data_test = list(data_train.items()), list(data_test.items())

    final = args.final
    test = args.test
    cross_validate = args.cross_validate
    record_runtime = True
    save_model = True
    if args.final:
        data_train = data_train + data_test
        test = False
        cross_validate = False
    experiment.train_all_nets(Models, data_train, data_test, epochs=args.epochs, batch_size=args.batch_size, kf_n_splits=10, 
                              cross_validate=cross_validate, test=test, save_model = save_model,
                              optimizer = args.opt, record_runtime=record_runtime, final=final)
