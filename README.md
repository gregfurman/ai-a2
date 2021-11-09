# AI Assignment 2

This repo contains information for the honours level AI course's assignment 2 codebase.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required libraries:

```bash
pip install transformers
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage
We used Google Colab for training and evaluation as GPUs offered a significant train and evaluation speedup. The link to the Colab notebook can be found [here](https://colab.research.google.com/drive/1mHQTh3EB2RfgMfvK6sxyCHm4drEmg0MG#scrollTo=TSve7a1a8ljr).
```bash

# Train and evaluate BERT model
python main.py --model_type "bert" --env "bert" --dataset "data/dataset.csv" --device "cuda" --batch_size 25 -lr 1e-5 --context_window 150 --max_iters 5 --num_workers 0 --evaluate True --train True

# Train and evaluate LSTM baseline model
python main.py  --model_type "lstm" --env "baseline" --dataset "data/dataset.csv" --device "cuda" --batch_size 25 -lr 1e-4 --hidden_size 512 --embed_dim 512 --n_layer 3 --max_iters 50 --num_workers 0 --evaluate True --train True

```

## Training & Evaluation

Training and evaluation can be done independently or together. If together, evaluation on the test set occurs following the number of training iterations reaching the maximum iteration flag ````--max_iters````.

### Train
In order to train a model, add the ```--train True``` flag to the ````main.py````

### Evaluation
In order to evaluate a model, add the ```--evaluate True``` flag to the ````main.py````

## Saving
### Training
All training logs are written to the ````/training_logs/```` directory

### Saving models
All saved models are written to the ````/saved_models/```` directory in a ```.pt``` format

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
