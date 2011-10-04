#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
ncores=$(seq 1 8)
#ncores="8"
#verif="1"
niter="30"
version=$(date +%s)
out="$HOME/res/plasma-gemm-$version.txt"

ninputs="$(seq 64 64 10240)"

# FLOAT
for n in $ninputs
do
	for p in $ncores
	do
		for i in `seq 1 $niter`
		do
		echo "./sgemm_matprod $n $p $verif"
		./sgemm_matprod $n $p $verif >> $out
		done
	done
done

# DOUBLE
for n in $ninputs
do
	for p in $ncores
	do
		for i in `seq 1 $niter`
		do
		echo "./dgemm_matprod $n $p $verif"
		./dgemm_matprod $n $p $verif >> $out
		done
	done
done
