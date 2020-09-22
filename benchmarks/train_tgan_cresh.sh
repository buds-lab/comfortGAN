#!/bin/sh
rm -dR output/ 
python train_tgan.py config_tgan_cresh.json 
rm -dR output/

