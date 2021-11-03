import torch
import torch.nn as nn
from transformers import  BertTokenizer, BertModel

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import dataset
# from dataset import data
# from sklearn.metrics import f1_score,classification_report,confusion_matrix


class BertClassifier(nn.Module):
    """
    BERT module for classification task.
    :param vocab_size: size of the vocab 
    :param hidden_size: embedding dimension
    :param context_window: the number of tokens to be used as context for question-answering
    """


    def __init__(
        self,      
        num_outputs,   
        context_window=200,
    ):
        super(BertClassifier,self).__init__()

        self.classes = num_outputs

        self.context_window = context_window

        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.context_window = context_window
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_layer = nn.Linear(self.model.config.hidden_size, self.classes)
    
    def forward(self, context):
        """
        Forward method for QA model.
        :param prompt_questions: list of questions and text pairs used for question-answering.  
        """
        
        encoding = self.tokenizer(
            context, 
            max_length=self.context_window, 
            truncation=True,padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )

        output = self.model(input_ids=encoding['input_ids'].to(device=self.device),
                attention_mask=encoding['attention_mask'].to(device=self.device))

        logits = self.output_layer(output["pooler_output"].to(device=self.device))

        return logits
    
    def predict(self, context):
        """
        Prediction method for classification task. 
        :param context: list of context strings to-be classified
        :returns: index from vocab of predicted class.
        """

        answer_preds = self.forward(context)
        return torch.argmax(answer_preds[-1])


class Baseline(nn.Module):
    
    def __init__(self, num_outputs, embedding_size, hidden_size, vocab, num_layers):
        super(Baseline, self).__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

       
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.number_of_layers = num_layers
        self.classes = num_outputs
       
        self.embedding = nn.Embedding(vocab, embedding_size).to(self.device)

        self.lstm = nn.LSTM(self.embedding_size,self.hidden_size,self.number_of_layers,batch_first=True)

        self.output_layer = nn.Linear(hidden_size,self.classes)        

    def forward(self, input_data):
       
        embeddings = self.embedding(input_data)

        out, _ = self.lstm(embeddings)
        out = out[:,-1,:]
        logits = self.output_layer(out)

        return logits


def train(filename,batch_size,embedding_size,hidden_size,number_of_layers,learning_rate):
    train_data = data(filename)
    train_dataloader = DataLoader(train_data,batch_size,collate_fn=train_data.custom_collate)

    model = Baseline(embedding_size,hidden_size,len(train_data.vocab),number_of_layers,len(train_data.label_to_id))
    optimizer = optim.Adam(model.parameters(),learning_rate)
    for batch_sentences,batch_labels in train_dataloader:
        
        logits = model(batch_sentences)
        loss = model.loss_function(logits,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss:",loss.item())

    return model

def eval(filename,batch_size,model):
    test_data = data(filename)
    test_dataloader = DataLoader(test_data,batch_size,collate_fn=test_data.custom_collate)
    y_predictions = []
    y_truths = []
    for batch_sentences,batch_labels in test_dataloader:
        logits = model(batch_sentences)
        y_pred = torch.argmax(logits,dim=-1)
        y_predictions.extend(list(y_pred.cpu().numpy()))
        y_truths.extend(list(batch_labels.cpu().numpy()))
    
    print(classification_report(y_truths,y_predictions,target_names=[{value : key for (key, value) in test_data.label_to_id.items()}[i] for i in range(len(test_data.label_to_id))]))

if __name__ == "__main__":
    train_data = data("data/train.csv")
    train_dataloader = DataLoader(train_data,64,collate_fn=train_data.custom_collate)

    model = Baseline(200,200,len(train_data.vocab),1,len(train_data.label_to_id))

    eval("data/train.csv",64,model)