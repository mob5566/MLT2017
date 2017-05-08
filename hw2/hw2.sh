#!/usr/bin/env sh

echo "MLT Homework 2\n"

# Get data from course website
wget https://www.csie.ntu.edu.tw/~htlin/course/mltech17spring/hw2/hw2_lssvm_all.dat

python2 q11_q12.py hw2_lssvm_all.dat
python2 q13_q14.py hw2_lssvm_all.dat
python2 q15_q16.py hw2_lssvm_all.dat
