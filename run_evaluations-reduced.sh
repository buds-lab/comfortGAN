#!/bin/sh

# batch_size n_critic latent dim

echo "[Start] Evaluation trials 64-1-100"
python evaluation_trials-reduced.py -a configs/config_comfortgan_ashrae-reduced_64-1-100.json 

echo "[Start] Evaluation trials 128-1-20"
python evaluation_trials-reduced.py -o configs/config_comfortgan_occutherm-reduced_128-1-20.json 
