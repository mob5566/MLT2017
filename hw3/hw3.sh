#!/usr/bin/env sh

echo "MLT Homework 3\n"

# Get data from course website
wget https://www.csie.ntu.edu.tw/~htlin/course/mltech17spring/hw3/hw3_train.dat
wget https://www.csie.ntu.edu.tw/~htlin/course/mltech17spring/hw3/hw3_test.dat

python2.7 q7_q13.py hw3_train.dat hw3_test.dat
python2.7 q14_q16.py hw3_train.dat hw3_test.dat
dot -Tpng Q14.dot > Q14.png
