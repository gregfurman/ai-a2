import argparse
import random
import torch

from model import BertClassifier, Baseline
from trainer import ClassificationModelTrainer
from dataset import dataBERT, data
from torch.utils.data import DataLoader

def experiment(
    exp_prefix,
    variant
):
    """
    Method for running QA model experiment.
    """
    device = variant.get('device', 'cuda') # cuda or cpu

    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, filename = variant['env'], variant['dataset'] 
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{filename}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='ai-assignment-2',
            config=variant
        )
    
    
    if model_type == "bert":
        dataset = dataBERT(filename)
        collate_fn = None
    elif model_type == "lstm":
        dataset = data(filename)
        collate_fn = dataset.custom_collate
    else:
        raise NotImplementedError

    vocab_size = len(dataset.vocab)
    output_size = len(dataset.label_to_id)

    if model_type == "bert":

        model = BertClassifier(
            hidden_size=variant["hidden_size"],
            num_outputs=output_size,
            context_window=variant["context_window"],
        )
    elif model_type == "lstm":

        model = Baseline(
            num_outputs=output_size,
            hidden_size=variant["hidden_size"],
            vocab=vocab_size, 
            num_layers=variant["n_layers"], 
            embedding_size=variant["embed_dim"],
        )

    model = model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    trainer = ClassificationModelTrainer(
        model,
        optimizer,
        loss_fn, 
        dataset=dataset,
        num_workers=variant["num_workers"],
        batch_size = variant['batch_size']
    )
    
    epochs = variant['max_iters']

    print("========== Beginning Training ==========\n")
    with open(f"./training_logs/{env_name}.json","w") as training_logs:

        max_accuracy = 0
        for iter in range(epochs):
            outputs = trainer.train_iteration(iter_num=iter+1, print_logs=True)
            
            if outputs['evaluation/accuracy'] >= max_accuracy:
                max_accuracy = outputs['evaluation/accuracy']
                torch.save(model,f"{variant['model_out']}/{variant['env']}.pt")
            
            if log_to_wandb:
                wandb.log(outputs)
            
            outputs["iteration"] = iter+1
            print(json.dumps(outputs),file=training_logs)
            training_logs.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='random_rollout')
    parser.add_argument('--dataset', type=str, default='data/Train.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # bert for BERT model, lstm for LSTM baseline
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool)
    parser.add_argument('--context_window', '-state', type=int, default=170) # size of context window for qa, or size of each state for dt
    parser.add_argument('--vocab_size', '-vocab', type=int, default=5)
    parser.add_argument('--model_out', type=str, default="./saved_models")
    parser.add_argument('--num_workers', type=int, default=2) # workers for dataloader

    args = vars(parser.parse_args())

    model_type = args["model_type"]

    experiment(f'{model_type}-experiment', variant=args)
