#!/bin/sh
rm -dR output/
python train_tgan.py config_tgan_occutherm-reduced.json 
rm -dR output/

