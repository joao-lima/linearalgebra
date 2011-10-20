#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
#verif="1"
niter="30"
version=$(date +%s)
out="$HOME/res/cublas-gemm-$version.txt"

# FLOAT
ninputs="$(seq 256 256 10240)"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgemm_matprod $n $verif"
	./sgemm_matprod $n $verif >> $out
	done
done

# DOUBLE
ninputs="$(seq 256 256 10240)"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgemm_matprod $n $verif"
	./dgemm_matprod $n $verif >> $out
	done
done
