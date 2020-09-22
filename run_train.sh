#!/bin/sh

# batch_size n_critic latent dim

echo "[Start] Train 64-1-100"
python train.py configs/config_comfortgan_ashrae-64-1-100.json 

echo "[Start] Train 128-1-20"
python train.py configs/config_comfortgan_occutherm-128-1-20.json 

echo "[Start] Train 128-3-80"
python train.py configs/config_comfortgan_cresh_128-3-80.json     
