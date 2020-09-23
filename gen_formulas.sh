#!/bin/bash

mkdir formulas
for s in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  randltl -n -1 a b c -r --ltl-priorities 'X=0, W=0, M=0' --tree-size=$s | ltlfilt --nnf -r --remove-wm --size=$s -n 1000 > formulas/$s.txt
done
