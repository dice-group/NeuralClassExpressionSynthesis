import numpy as np, copy
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.utils import resample
from torch.utils.data import DataLoader
import sys, os, json
base_path = os.path.dirname(os.path.realpath(__file__)).split('utils')[0]
sys.path.append(base_path)
from .base import DataLoaderBase
from .dataloader import NCESDataLoader
from nces.synthesizer import ConceptSynthesizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss, functional as F
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import time

class Experiment:
    
    def __init__(self, kwargs):
        self.decay_rate = kwargs.decay_rate
        self.clip_value = kwargs.grad_clip_value
        setattr(kwargs, "pretrained_concept_synthesizer", base_path+"datasets/"+"carcinogenesis/Model_weights/"+kwargs.learner_name+".pt") # default pretrained path
        self.num_workers = kwargs.num_workers
        self.load_pretrained = kwargs.load_pretrained
        self.cs = ConceptSynthesizer(kwargs)
        self.kwargs = kwargs
        
    
    def before_pad(self, arg):
        arg_temp = []
        for atm in arg:
            if atm == 'PAD':
                break
            arg_temp.append(atm)
        return arg_temp
    
    
    def compute_accuracy(self, prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(self.before_pad(DataLoaderBase.decompose(arg1_)))
            else:
                arg1_ = set(self.before_pad(arg1_))
            if isinstance(arg2_, str):
                arg2_ = set(self.before_pad(DataLoaderBase.decompose(arg2_)))
            else:
                arg2_ = set(self.before_pad(arg2_))
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = self.before_pad(DataLoaderBase.decompose(arg1_))
            else:
                arg1_ = self.before_pad(arg1_)
            if isinstance(arg2_, str):
                arg2_ = self.before_pad(DataLoaderBase.decompose(arg2_))
            else:
                arg2_ = self.before_pad(arg2_)
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
        size = sum([p.numel() for p in self.cs.model.parameters()])
        size_ = 0
        print("Synthesizer: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        return size
        
    def collate_batch(self, batch):
        pos_emb_list = []
        neg_emb_list = []
        target_labels = []
        for pos_emb, neg_emb, label in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
            target_labels.append(label)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels    
        
        
    def map_to_token(self, idx_array):
        return self.cs.model.inv_vocab[idx_array]
    
    def train(self, train_dataloader, test_dataloader, epochs=200, test=False, save_model=False, kb_emb_model="ConEx", optimizer='Adam', record_runtime=False, final=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_size = self.show_num_learnable_params()
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.cs.model.name, self.kwargs.knowledge_base_path.split("/")[-2]))
        print("#"*50, "\n")
        if self.load_pretrained:
            try:
                self.cs.load_pretrained()
                print("\nUsing pretrained model...\n")
            except Exception:
                print("\n")
                print("**** Could not load from pretrained, missing file ****\n")
        synthesizer = copy.deepcopy(self.cs.model)
        desc = kb_emb_model+'_'+synthesizer.name
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
            if Train_acc['hard'] and Train_acc['hard'][-1] > best_score:
                best_score = Train_acc['hard'][-1]
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
            results_dict.update({"Model size": model_size, "Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
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
                    
    def train_all_nets(self, List_nets, train_data, test_data, epochs=200, batch_size=64, test=False, save_model = False, kb_emb_model='ConEx', optimizer = 'Adam', record_runtime=False, final=False):
        self.embeddings = self.cs.get_embedding()
        train_dataset = NCESDataLoader(train_data, self.embeddings, self.kwargs)
        self.num_examples = train_dataset.num_examples
        print(f"\n***Number of examples per learning problem: {self.num_examples}***\n")
        test_dataset = NCESDataLoader(test_data, self.embeddings, self.kwargs)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
        
        if not os.path.exists(base_path+f"datasets/{self.kb}/Training_curves/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Training_curves/")
        if not os.path.exists(base_path+f"datasets/{self.kb}/Plot_data/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Plot_data/")
        for net in List_nets:
            self.cs.learner_name = net
            print()
            print('Learner: ', self.cs.learner_name)
            self.cs.refresh()
            train_soft_acc, train_hard_acc, train_l = self.train(train_dataloader, test_dataloader, epochs, test, save_model, kb_emb_model, optimizer, record_runtime, final)
            with open(base_path+f"datasets/{self.kb}/Plot_data/{kb_emb_model}_{net}_plot_data.json", "w") as plot_file:
                json.dump({"soft acc": list(train_soft_acc), "hard acc": list(train_hard_acc), "loss": list(train_l)}, plot_file, indent=3)

            
