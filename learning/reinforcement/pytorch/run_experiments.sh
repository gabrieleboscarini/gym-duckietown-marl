#!/bin/bash

# Script to reproduce results

for ((i=10;i<16;i+=1))
do 
	python train_multibot.py \
	--seed $i
done