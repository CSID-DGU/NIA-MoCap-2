#!/bin/sh
path='/home/irteam/bvh'
files=$(find $path/ -name "*.bvh")

for file in $files; do
   python __main__.py $file
done
