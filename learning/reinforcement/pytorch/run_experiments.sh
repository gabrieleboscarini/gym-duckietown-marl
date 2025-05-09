#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python train_TD3.py \
	--seed $i
done