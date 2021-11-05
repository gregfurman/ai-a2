import os 
import pandas
from torch.utils.data import Dataset, DataLoader
import re
import copy
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class data(Dataset):

    def __init__(self, filename):
        self.sentences, self.labels, self.vocab, self.label_to_id = self.read_file(filename)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
       
    def read_file(self,filename):

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, filename)

        df = pandas.read_csv(filename)
        
        vocab = {}
    
        sentences = []
        sen = []
        current_id = 0
       
        for line in df["tweet"]:
            sen = re.sub(r'[^\w\s]','',line).lower().split(" ")
            sentences.append(sen)
            for word in sen:
               
                if not (word in vocab):
                    vocab[word] = current_id
                    current_id += 1

        labels = list(df["type"])
        labels_id = {}
        current_id = 0
        for label in labels:
            if not (label in labels_id):
                    labels_id[label] = current_id
                    current_id += 1
        

        return sentences, labels, vocab, labels_id


    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):
        
        word_ids = torch.tensor(list(map(lambda x: self.vocab[x],self.sentences[index])),device=self.device)
        label_ids = torch.tensor(self.label_to_id[self.labels[index]],device=self.device)
        
        return word_ids, label_ids



    def custom_collate(self,batch):
        data = [item[0] for item in batch]
        data = pad_sequence(data, batch_first=True)
        targets = [item[1] for item in batch]
        return [data, torch.stack(targets)]

class dataBERT(data):
    
    def __init__(self, filename):
        
        super().__init__(filename)
    
    def __getitem__(self, index):
        
        # label_ids = torch.tensor(self.label_to_id[self.labels[index]])
        return " ".join(self.sentences[index]), self.label_to_id[self.labels[index]]

