#!/bin/bash

# Script to reproduce results

for ((i=12;i<17;i+=1))
do 
	python train_multibot.py \
	--seed $i
done