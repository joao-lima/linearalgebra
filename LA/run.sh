#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
#verif="1"
niter="1"
version=$(date +%s)
out="$HOME/res/gemm-$version.txt"

ninputs="$(seq 64 64 10240)"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgemm_matprod $n"
	./sgemm_matprod $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgemm_pt_matprod $n"
	./sgemm_pt_matprod $n >> $out
	done
done


# DOUBLE
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgemm_matprod $n"
	./dgemm_matprod $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgemm_pt_matprod $n"
	./dgemm_pt_matprod $n >> $out
	done
done
