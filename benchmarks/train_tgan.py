import sys
import time
import json
import pandas as pd
from tgan.model import TGANModel

with open(str(sys.argv[1]), 'r') as f:
    config = json.load(f)

df_train = pd.read_pickle(config['df_train']) 
cont_columns = config['continuous_cols']

tgan = TGANModel(
    cont_columns,
    batch_size = config['batch_size'],
    z_dim = config['z_dim'],
    num_gen_rnn = config['num_gen_rnn'],
    num_gen_feature = config['num_gen_feature'],
    num_dis_layers = config['num_dis_layers'],
    num_dis_hidden = config['num_dis_hidden'],
    learning_rate = config['learning_rate'],
    noise = config['noise'],
    max_epoch = config['max_epoch'],
    steps_per_epoch = config['steps_per_epoch']
    )

model_path = config['model_path']

start_time = time.time()
# fit the TGAN
tgan.fit(df_train)
print("--- %s seconds ---" % (time.time() - start_time))

tgan.save(model_path, force=True)
