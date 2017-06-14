#!/usr/bin/env sh

echo "MLT Homework 4\n"

# Get data from course website
wget https://www.csie.ntu.edu.tw/~htlin/course/mltech17spring/hw3/hw3_train.dat
wget https://www.csie.ntu.edu.tw/~htlin/course/mltech17spring/hw3/hw3_test.dat

python2.7 hw4_questions.py hw3_train.dat hw3_test.dat
