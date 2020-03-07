#!/bin/sh
# unzip the train and test datasets

#create the directories for the images
mkdir -p Data
mkdir -p Data/Train
mkdir -p Data/Test

# create workdirectory - where the program saves the weights, etc
mkdir -p wdir

# extract the images
rar x Train.rar Data/Train
rar x Test.rar Data/Test
