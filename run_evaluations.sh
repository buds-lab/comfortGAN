#!/bin/sh

# batch_size n_critic latent dim

echo "[Start] Evaluation trials 64-1-100"
python evaluation_trials.py -a configs/config_comfortgan_ashrae_64-1-100.json 

echo "[Start] Evaluation trials 128-1-20"
python evaluation_trials.py configs/config_comfortgan_occutherm_128-1-20.json configs/config_comfortgan_cresh_128-1-20.json

echo "[Start] Evaluation trials 128-3-80"
python evaluation_trials.py configs/config_comfortgan_occutherm_128-3-80.json configs/config_comfortgan_cresh_128-3-80.json
