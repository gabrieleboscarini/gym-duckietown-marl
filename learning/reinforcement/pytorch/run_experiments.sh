#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
	python train_multibot.py \
	--seed $i
done