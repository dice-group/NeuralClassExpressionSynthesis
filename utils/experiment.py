import numpy as np, copy
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.utils import resample
from torch.utils.data import DataLoader
import sys, os, json
base_path = os.path.dirname(os.path.realpath(__file__)).split('utils')[0]
sys.path.append(base_path)
from nces import BaseConceptSynthesis
from .dataloader import CSDataLoader
from nces.synthesizer import ConceptSynthesizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import f1_score, accuracy_score
import time

class Experiment:
    
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.decay_rate = kwargs.decay_rate
        self.clip_value = kwargs.grad_clip_value
        self.cs = ConceptSynthesizer(kwargs)
        self.num_workers = kwargs.num_workers
            
    def compute_accuracy(self, prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(BaseConceptSynthesis.decompose(arg1_)) - {'PAD'}
            else:
                arg1_ = set(arg1_)
            if isinstance(arg2_, str):
                arg2_ = set(BaseConceptSynthesis.decompose(arg2_))
            else:
                arg2_ = set(arg2_)
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))
        
        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = [atm for atm in BaseConceptSynthesis.decompose(arg1_) if atm != 'PAD']
            else:
                arg1_ = [atm for atm in arg1_ if atm != 'PAD']
            if isinstance(arg2_, str):
                arg2_ = BaseConceptSynthesis.decompose(arg2_)
            else:
                arg2_ = [atm for atm in arg2_ if atm != 'PAD']
            return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc
          
            
    def get_optimizer(self, synthesizer, optimizer='Adam'):
        if optimizer == 'Adam':
            return torch.optim.Adam(synthesizer.parameters(), lr=self.kwargs.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(synthesizer.parameters(), lr=self.kwargs.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(synthesizer.parameters(), lr=self.kwargs.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')
    
    def show_num_learnable_params(self):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in self.cs.synthesizer.parameters()])
        size_ = 0
        print("Synthesizer: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        
    @staticmethod
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
        #target_labels = pad_sequence(target_labels, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels
            
        
    def map_to_token(self, idx_array):
        return self.cs.synthesizer.inv_vocab[idx_array]
    #[self.cs.synthesizer.inv_vocab[idx] for idx in idx_array if idx != -100]
    
    def train(self, train_dataloader, test_dataloader, epochs=200, kf_n_splits=10, test=False, save_model = False, optimizer = 'Adam', record_runtime=False, final=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.show_num_learnable_params()
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.cs.synthesizer.name, self.kwargs.knowledge_base_path.split("/")[-2]))
        print("#"*50, "\n")
        synthesizer = copy.deepcopy(self.cs.synthesizer)
        desc = synthesizer.name
        if final:
            desc = desc+'_final'
        if train_on_gpu:
            synthesizer.cuda()
        opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        Train_acc = defaultdict(list)
        best_score = 0.
        if record_runtime:
            t0 = time.time()
        for e in range(epochs):
            soft_acc, hard_acc = [], []
            train_losses = []
            for x1, x2, labels in tqdm(train_dataloader):
                target_sequence = self.map_to_token(labels)
                if(train_on_gpu):
                    #x, y_numerical = x.cuda(), y_numerical.cuda()
                    x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                pred_sequence, scores = synthesizer(x1, x2)
                cs_loss = synthesizer.loss(scores, labels)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
                train_losses.append(cs_loss.item())
                opt.zero_grad()
                cs_loss.backward()
                clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                opt.step()
                if self.decay_rate:
                    self.scheduler.step()
            tr_soft_acc, tr_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            Train_loss.append(np.mean(train_losses))
            Train_acc['soft'].append(tr_soft_acc)
            Train_acc['hard'].append(tr_hard_acc)
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Train loss: {:.4f}...".format(np.mean(train_losses)),
                  "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                  "Train hard acc: {:.2f}%...".format(tr_hard_acc))
            if np.random.rand() > 0.7:
                print("Visualizing some prediction: ", pred_sequence[np.random.choice(range(x1.shape[0]))])
                print()
            weights_cs = copy.deepcopy(synthesizer.state_dict())
            if Train_acc['soft'] and Train_acc['soft'][-1] > best_score:
                best_score = Train_acc['soft'][-1]
                best_weights_cs = weights_cs
        synthesizer.load_state_dict(best_weights_cs)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Concept synthesizer": synthesizer.name,
                           "Number of Epochs": epochs, "Runtime (s)": duration}
            if not os.path.exists(base_path+f"datasets/{self.kb}/Runtime/"):
                os.mkdir(base_path+f"datasets/{self.kb}/Runtime")
            with open(base_path+f"datasets/{self.kb}/Runtime/"+"Runtime_"+desc+".json", "w") as file:
                json.dump(runtime_info, file, indent=3)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            soft_acc, hard_acc = [], []
            for x1, x2, labels in test_dataloader:
                if train_on_gpu:
                    x1, x2 = x1.cuda(), x2.cuda()
                pred_sequence, _ = synthesizer(x1, x2)
                target_sequence = target_sequence = self.map_to_token(labels)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        print("Train soft accuracy: {} ... Train hard accuracy: {}".format(max(Train_acc['soft']), max(Train_acc['hard'])))
        print()
        results_dict.update({"Train Max Soft Acc": max(Train_acc['soft']), "Train Max Hard Acc": max(Train_acc['hard']), "Train Min Loss": min(Train_loss)})
        if not os.path.exists(base_path+f"datasets/{self.kb}/Results/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Results/")
        with open(base_path+f"datasets/{self.kb}/Results/"+"Train_Results_"+desc+".json", "w") as file:
                json.dump(results_dict, file, indent=3)
        if save_model:
            if not os.path.exists(base_path+f"datasets/{self.kb}/Model_weights/"):
                os.mkdir(base_path+f"datasets/{self.kb}/Model_weights/")
            torch.save(synthesizer, base_path+f"datasets/{self.kb}/Model_weights/"+desc+".pt")
            print("{} saved".format(synthesizer.name))
            print()
        plot_data = (np.array(Train_acc['soft']), np.array(Train_acc['hard']), Train_loss)
        return plot_data
        
    
    def cross_validate(self, train_data, test_dataloader, epochs=200, batch_size=64, kf_n_splits=10, test=False, save_model = False, optimizer = 'Adam', *kwargs):            
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")   
        best_score = 0.
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.cs.synthesizer.name, self.kb))
        print("#"*50, "\n")
        from sklearn.model_selection import KFold
        Kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=142)
        fold = 0
        All_losses = defaultdict(list)
        All_acc = defaultdict(list)
        best_val_score = 0.
        iterable = list(range(len(train_data)))
        for train_index, valid_index in Kf.split(iterable):
            self.show_num_learnable_params()
            synthesizer = copy.deepcopy(self.cs.synthesizer)
            if train_on_gpu:
                synthesizer.cuda()
            opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer)
            if self.decay_rate:
                self.scheduler = ExponentialLR(opt, self.decay_rate)
            train_data = np.array(train_data, dtype=object)
            train_dataset = CSDataLoader(train_data[train_index], self.embeddings, self.kwargs)
            valid_dataset = CSDataLoader(train_data[valid_index], self.embeddings, self.kwargs)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
            fold += 1
            print("*"*50)
            print("Fold {}/{}:\n".format(fold, kf_n_splits))
            print("*"*50, "\n")
            Train_losses = []
            Val_losses = []
            Train_acc = defaultdict(list)
            Val_acc = defaultdict(list)
            for e in range(epochs):
                soft_acc, hard_acc = [], []
                train_losses = []
                for x1, x2, labels in tqdm(train_dataloader):
                    target_sequence = self.map_to_token(labels)
                    if(train_on_gpu):
                        x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                    pred_sequence, scores = synthesizer(x1, x2)
                    cs_loss = synthesizer.loss(scores, y_numerical)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
                    train_losses.append(cs_loss.item())
                    opt.zero_grad()
                    cs_loss.backward()
                    clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                    opt.step()
                    if self.decay_rate:
                        self.scheduler.step()
                tr_soft_acc, tr_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                # Get validation loss
                val_losses = []
                synthesizer.eval()
                soft_acc, hard_acc = [], []
                for x1, x2, labels in valid_dataloader:
                    target_sequence = self.map_to_token(labels)
                    if(train_on_gpu):
                        x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                    pred_sequence, scores = synthesizer(x1, x2)
                    val_loss = synthesizer.loss(scores, labels)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
                    val_losses.append(val_loss.item())
                val_soft_acc, val_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                synthesizer.train() # reset to train mode after iteration through validation data
                Train_losses.append(np.mean(train_losses))
                Val_losses.append(np.mean(val_losses))
                Val_acc['soft'].append(val_soft_acc)
                Val_acc['hard'].append(val_hard_acc)
                Train_acc['soft'].append(tr_soft_acc)
                Train_acc['hard'].append(tr_hard_acc)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Train loss: {:.4f}...".format(np.mean(train_losses)),
                      "Val loss: {:.4f}...".format(np.mean(val_losses)),
                      "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                      "Train hard acc: {:.2f}%...".format(tr_hard_acc),
                      "Val soft acc: {:.2f}%...".format(val_soft_acc),
                      "Val hard acc: {:.2f}%".format(val_hard_acc))
                weights_cs = copy.deepcopy(synthesizer.state_dict())
                if Val_acc['soft'] and max(Val_acc['soft']) > best_val_score:
                    best_val_score = max(Val_acc['soft'])
                    best_weights_cs = weights_cs
                All_losses["train"].append(Train_losses)
                All_losses["val"].append(Val_losses)
                All_acc["train"].append(Train_acc)
                All_acc["val"].append(Val_acc)
        synthesizer.load_state_dict(best_weights_cs)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            soft_acc, hard_acc = [], []
            for x1, x2, labels in test_dataloader:
                target_sequence = self.map_to_token(labels)
                if train_on_gpu:
                    x1, x2 = x1.cuda(), x2.cuda()
                pred_sequence, _ = synthesizer(x1, x2)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        plot_data = (np.array([a['soft'] for a in All_acc['train']]).mean(1), np.array([a['hard'] for a in All_acc['train']]).mean(1),
                     np.array([a['soft'] for a in All_acc['val']]).mean(1), np.array([a['hard'] for a in All_acc['val']]).mean(1),
                     np.array([l for l in All_losses['train']]).mean(1), np.array([l for l in All_losses['val']]).mean(1))
        print("Train soft accuracy: {} ... Train hard accuracy: {} ... Val soft accuracy: {} ... Val hard accuracy: {}".\
              format(max(plot_data[0]), max(plot_data[1]), max(plot_data[2]), max(plot_data[3])))
        print()
        results_dict.update({"Train Max Avg Soft Acc": max(plot_data[0]),
                              "Train Max Avg Hard Acc": max(plot_data[1]),
                              "Val Max Avg Soft Acc": max(plot_data[2]),
                              "Val Max Avg Hard Acc": max(plot_data[3]),
                              "Train Min Avg Loss": min(plot_data[4]),
                              "Val Min Avg Loss": min(plot_data[5])})
        if not os.path.exists(base_path+f"datasets/{self.kb}/Results/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Results/")
        with open(base_path+f"datasets/{self.kb}/Results/"+"Train_Results_"+synthesizer.name+".json", "w") as file:
                json.dump(results_dict, file, indent=3)

        if save_model:
            if not os.path.exists(base_path+f"datasets/{self.kb}/Model_weights/"):
                os.mkdir(base_path+f"datasets/{self.kb}/Model_weights/")
            torch.save(synthesizer, base_path+f"datasets/{self.kb}/Model_weights/"+synthesizer.name+".pt")
            print("{} saved".format(synthesizer.name))
        return plot_data
    
    
    def train_and_eval(self, train_data, test_data, epochs=200, batch_size=64, kf_n_splits=10, cross_validate=False, test=False, save_model = False, optimizer = 'Adam', record_runtime=False, final=False):
        
        """
        function for training a concept length learner in DL KBs
        
        key args
        -> cll_batch_size: batch_size for the concept learner training (cs: concept synthesizer)
        -> tc_batch_size: batch_size for the training of the embedding model (tc: triple classification)
        key args
        """
        if cross_validate:
            return self.cross_validate(train_data, test_data, epochs, batch_size,
                                       kf_n_splits, test, save_model, optimizer, record_runtime, final)

        else:
            return self.train(train_data, test_data, epochs,
                    kf_n_splits, test, save_model, optimizer, record_runtime, final)
            
            
    def train_all_nets(self, List_nets, train_data, test_data, epochs=200, batch_size=64, kf_n_splits=10, cross_validate=False, test=False, save_model = False, optimizer = 'Adam', record_runtime=False, final=False):
        self.embeddings = self.cs.get_embedding()
        train_dataset = CSDataLoader(train_data, self.embeddings, self.kwargs)
        test_dataset = CSDataLoader(test_data, self.embeddings, self.kwargs)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
        
        if not os.path.exists(base_path+f"datasets/{self.kb}/Training_curves/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Training_curves/")
        if not os.path.exists(base_path+f"datasets/{self.kb}/Plot_data/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Plot_data/")
        if cross_validate:
            for net in List_nets:
                self.cs.learner_name = net
                self.cs.refresh()
                train_soft_acc, train_hard_acc, val_soft_acc, val_hard_acc, train_l, val_l = self.train_and_eval(train_data, test_dataloader, epochs, batch_size, kf_n_splits, cross_validate, test, save_model, optimizer, record_runtime, final)
                with open(base_path+f"datasets/{self.kb}/Plot_data/{net}_plot_data_with_val.json", "w") as plot_file:
                    json.dump({'train': {"soft acc": list(train_soft_acc), "hard acc": list(train_hard_acc), "loss": list(train_l)},
                               'val': {"soft acc": list(val_soft_acc), "hard acc": list(val_hard_acc), "loss": list(val_l)}},
                                plot_file, indent=3)
                        
        else:
            for net in List_nets:
                self.cs.learner_name = net
                print()
                print('Learner: ', self.cs.learner_name)
                self.cs.refresh()
                train_soft_acc, train_hard_acc, train_l = self.train_and_eval(train_dataloader, test_dataloader, epochs, batch_size, kf_n_splits, cross_validate, test, save_model, optimizer, record_runtime, final)
                with open(base_path+f"datasets/{self.kb}/Plot_data/{net}_plot_data.json", "w") as plot_file:
                    json.dump({"soft acc": list(train_soft_acc), "hard acc": list(train_hard_acc), "loss": list(train_l)}, plot_file, indent=3)

            
