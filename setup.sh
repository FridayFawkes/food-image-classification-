#!/bin/sh
# unzip the train and test datasets

#create the directories for the images
mkdir -p data
mkdir -p data/train
mkdir -p data/test

# extract the images
rar x Train.rar data/train
rar x Test.rar data/test
