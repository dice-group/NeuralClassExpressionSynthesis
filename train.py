import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import os, json
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


from utils.experiment import Experiment
from utils.data import Data
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
parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'],
                    help='Knowledge base name')
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'LSTM', 'GRU'], help='Neural models')
parser.add_argument('--kb_emb_model', type=str, default='ConEx', help='Embedding model name')
parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load pretrained models')
parser.add_argument('--learner_name', type=str, default="SetTransformer", choices=['LSTM', 'GRU', 'SetTransformer'], help='Neural model')
parser.add_argument('--knowledge_base_path', type=str, default="", help='Path to KB owl file')
parser.add_argument('--path_to_csv_embeddings', type=str, default="", help='KB embedding path')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--embedding_dim', type=int, default=40, help='Number of embedding dimensions')
parser.add_argument('--input_size', type=int, default=40, help='Number of embedding dimensions in the input')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use to load training data')
parser.add_argument('--proj_dim', type=int, default=128, help='The projection dimension for examples')
parser.add_argument('--num_inds', type=int, default=32, help='Number of induced instances')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of seed components in the output')
parser.add_argument('--num_examples', type=int, default=1000, help='Total number of examples for concept learning')
parser.add_argument('--ln', type=str2bool, default=False, help='Whether to use layer normalization')
parser.add_argument('--decay_rate', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--grad_clip_value', type=float, default=5.0, help='Gradient clip value')
parser.add_argument('--opt', type=str, default='Adam', help='Name of the optimizer to use')
parser.add_argument('--rnn_n_layers', type=int, default=2, help='Number of recurrent network layers')
parser.add_argument('--max_length', type=int, default=48, help='Maximum length of class expressions')
parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout rate in neural networks')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--shuffle_examples', type=str2bool, default=False, help='Whether to shuffle positive and negative examples in the dataloader')
parser.add_argument('--test', type=str2bool, default=True, help='Whether to evaluate the concept synthesizer on the test data during training')
parser.add_argument('--final', type=str2bool, default=False, help='Whether to train the concept synthesizer on test+train data')
parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the model after training')

args = parser.parse_args()
    
print("Setting: ", vars(args))

for kb in args.kbs:
    data_train_path = f"datasets/{kb}/Train_data/Data.json"
    with open(data_train_path, "r") as file:
        data_train = json.load(file)

    data_test_path = f"datasets/{kb}/Test_data/Data.json"
    with open(data_test_path, "r") as file:
        data_test = json.load(file)
        
    args.knowledge_base_path = f"datasets/{kb}/{kb}.owl"
    args.path_to_csv_embeddings = f"datasets/{kb}/Embeddings/{args.kb_emb_model}_entity_embeddings.csv"
    experiment = Experiment(args)
    setattr(experiment, 'kb', kb)
    data_train, data_test = list(data_train.items()), list(data_test.items())

    final = args.final
    test = args.test
    if args.final:
        data_train = data_train + data_test
        test = False
    experiment.train_all_nets(args.models, data_train, data_test, epochs=args.epochs, 
                              batch_size=args.batch_size, test=test, 
                              save_model = args.save_model, kb_emb_model=args.kb_emb_model,
                              optimizer = args.opt, record_runtime=True, final=final)
