import time

import torch
from model import BertClassifier
from sklearn.metrics import f1_score,classification_report,confusion_matrix

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
        train_set,
        val_set,
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_set = train_set
        self.val_set = val_set

    def train_step(self):
        total_losses = []
        hits = 0
        for prompts, labels in self.train_set:

            output = self.model.forward(list(prompts))
            labels_tensor = torch.tensor(labels,device=self.model.device)
            loss = self.loss_fn(output,labels_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            total_losses.append(loss.detach().cpu().item()) 
            hits += (torch.argmax(output,dim=1) == labels_tensor).sum().detach()

        return total_losses, hits.cpu().item()/len(self.train_set)

    def eval(dataloader, model):
        y_predictions = []
        y_truths = []

        for prompts, labels in dataloader:

            labels_tensor = torch.tensor(labels,device=model.device)
            logits = model(list(prompts))

            y_pred = torch.argmax(logits,dim=-1)
            y_predictions.extend(list(y_pred.cpu().numpy()))
            y_truths.extend(list(batch_labels.cpu().numpy()))
        
        class_report = classification_report(
            y_truths,
            y_predictions,
            target_names=[{value : key for (key, value) in test_data.label_to_id.items()}[i] for i in range(len(test_data.label_to_id))],
        )

        print(class_report)
        
        return class_report

    def train_iteration(self, num_steps=0, iter_num=0, print_logs=False):


        logs = dict()

        train_start = time.time()

        self.model.train()

        train_losses, train_accuracy = self.train_step()
        if self.scheduler is not None:
            self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        self.model.eval()

        eval_start = time.time()

        # classification report on validation set
        self.eval(self.val_set, self.model)
        
        logs["evaluation/accuracy"] = hits.cpu().item()/len(self.val_subset)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/accuracy'] = train_accuracy

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

