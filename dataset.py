import os 
import pandas
from torch.utils.data import Dataset, DataLoader
import re
import copy
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from collections import defaultdict

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

class dataBERT(Dataset):
    
    def __init__(self, filename, window_length):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.sentences, self.labels, self.vocab, self.label_to_id = self.read_file(filename,window_length)

        
    def read_file(self,filename,window_length):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, filename)

        df = pandas.read_csv(filename)
        
        vocab = {}
        current_id = 0
       
        tokenized_sents = tokenizer(
            [tweet for tweet in df["tweet"]], 
            max_length=window_length, 
            truncation=True,padding='max_length',
            add_special_tokens=True,
        )

        sentences = [{'input_ids':input_id, 'attention_mask':att_mask }  for input_id, att_mask in zip(tokenized_sents['input_ids'],tokenized_sents['attention_mask'])]

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
        
        # label_ids = torch.tensor(self.label_to_id[self.labels[index]])
        return self.sentences[index], self.label_to_id[self.labels[index]]

    def custom_collate(self,batch):
        targets = []      

        collated_encodings = defaultdict(list)

        for encoding,label in batch:
            for k, v in encoding.items():
                collated_encodings[k].append(v)
            targets.append(label)
        
        collated_encodings['input_ids'] = torch.tensor(collated_encodings['input_ids']).to(device=self.device)
        collated_encodings['attention_mask'] = torch.tensor(collated_encodings['attention_mask']).to(device=self.device)

        return [collated_encodings , torch.tensor(targets).to(device=self.device)]