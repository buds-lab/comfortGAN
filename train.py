import sys
import json
import torch
import pandas as pd
from utils import *
from comfortgan import *

# load config file from CLI
with open(str(sys.argv[1]), 'r') as f:
    config = json.load(f)

# load data
df_train = pd.read_pickle(config['df_train']) 
cat_cols = config['categorical_cols']

# extract parameters from config file
experiment_name = config['name']
dataset_name = config['df_name']
BATCH_SIZE = config['batch_size']
gamma = config['gamma']
scaling = config['scaling']
latent_dim = config['z_dim']
n_critic = config['n_critic']
seed = config['seed']
max_itr = config['max_itr']

BUFFER_SIZE = df_train.shape[0] # same size as dataset means uniform sampling
columns_names = df_train.columns.values

# transform data
X_encoded, scaler, ohe_cat, ohe_label, list_label, num_cont = data_transform(df_train, cat_cols=cat_cols, scaling=scaling, gamma=gamma)
X_train = torch.utils.data.DataLoader(X_encoded, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed)
data_dim = X_encoded.shape[1] - len(list_label) # susbract the columns from the one-hot-encoded label

max_itr = 20000 
if dataset_name == "ashrae":
    max_itr = 200000 # for ashrae, dataset is way bigger now
#elif dataset_name == "cresh":
#    max_itr = 15000 # cresh seems to require more steps to converge
#
# G is trained for max_itr, D is trained for max_itr * n_critic
EPOCHS = max(max_itr * n_critic // (len(X_train)), 1) 
print("Epochs: {}".format(EPOCHS))

# initialize gan
comfortgan = comfortGAN(data_dim, 
                        latent_dim, 
                        gamma, 
                        list_label,
                        num_cont, 
                        columns_names, 
                        cat_cols, 
                        scaler, 
                        ohe_cat, 
                        ohe_label)

run_name = experiment_name + "-" + dataset_name
comfortgan.train(X_train, EPOCHS, n_critic, run_name=run_name, log=True)

# model_path = "models/comfortGAN-" + experiment_name + "-" + dataset_name + ".pkl"
model_path = config['model_path']
torch.save(comfortgan, model_path)
print("Model saved!")
