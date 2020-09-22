#!/bin/sh
rm -dR output/
python train_tgan.py config_tgan_ashrae-reduced.json 
rm -dR output/

