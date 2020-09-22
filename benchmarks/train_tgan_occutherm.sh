#!/bin/sh
rm -dR output/
python train_tgan.py config_tgan_occutherm.json 
rm -dR output/

