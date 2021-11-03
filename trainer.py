import time

import torch
import numpy as np
from model import BertClassifier
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from torch.utils.data import DataLoader

class ClassificationModelTrainer:
    """
    Trainer class for classification model.
    
    :param model: model to be trained
    :param optimizer: optimizer to be used for training. Weights and decay determined before passing
    :param loss_fn: loss function to be used
    :param filename: name of the dataset with which to load data from
    :param num_workers: number of workers used for dataloader
    """
    def __init__(
        self,
        model, 
        optimizer,
        loss_fn,
        dataset,
        **kwargs
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # calculates the training and validation set sizes.
        # context_window size used as random seed for splitting.
        train_size = round(len(dataset)*0.4)
        val_size = len(dataset) - train_size
        test_size = round(len(dataset)*0.4)

        train_subset, val_subset, test_set = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        

        self.train_set = DataLoader(dataset=train_subset, shuffle=True, **kwargs)
        self.val_set = DataLoader(dataset=val_subset, shuffle=False, **kwargs)
        self.test_set = DataLoader(dataset=test_set, shuffle=False, **kwargs)

        self._target_names = [{value : key for (key, value) in dataset.label_to_id.items()}[i] for i in range(len(dataset.label_to_id))]

        self.start_time = time.time()

    def train_step(self):
        total_losses = []
        for prompts, labels in self.train_set:

            output = self.model.forward(list(prompts))
            labels_tensor = torch.tensor(labels,device=self.model.device)
            loss = self.loss_fn(output,labels_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            total_losses.append(loss.detach().cpu().item()) 

        return total_losses

    def eval(self, dataloader, model):
        y_predictions = []
        y_truths = []

        for prompts, labels in dataloader:

            labels_tensor = torch.tensor(labels,device=model.device)
            logits = model(list(prompts))

            y_pred = torch.argmax(logits,dim=-1)
            y_predictions.extend(list(y_pred.cpu().numpy()))
            y_truths.extend(list(labels_tensor.cpu().numpy()))
        
        class_report = classification_report(
            y_truths,
            y_predictions,
            target_names=self._target_names,
            output_dict=True
        )

        return class_report

    def train_iteration(self, num_steps=0, iter_num=0, print_logs=False):


        logs = dict()

        train_start = time.time()

        self.model.train()

        train_losses = self.train_step()

        logs['time/training'] = time.time() - train_start

        self.model.eval()

        eval_start = time.time()

        # classification report on validation set
        val_accuracy = self.eval(self.val_set, self.model)["macro avg"]
        train_accuracy = self.eval(self.train_set, self.model)["macro avg"]

        logs["evaluation/accuracy"] = val_accuracy

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/accuracy'] = train_accuracy


        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

