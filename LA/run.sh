#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
#verif="1"
niter="30"
version=$(date +%s)
SCRATCH=$HOME
ninputs="$(seq 256 256 10240)"

out="$SCRATCH/res/atlas-potrf-$version.txt"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./spotrf_matcholesky $n"
	./spotrf_matcholesky $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dpotrf_matcholesky $n"
	./dpotrf_matcholesky $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./spotrf_pt_matcholesky $n"
	./spotrf_pt_matcholesky $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dpotrf_pt_matcholesky $n"
	./dpotrf_pt_matcholesky $n >> $out
	done
done

# matlu
out="$SCRATCH/res/atlas-getrf-$version.txt"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgetrf_matlu $n"
	./sgetrf_matlu $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgetrf_matlu $n"
	./dgetrf_matlu $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgetrf_pt_matlu $n"
	./sgetrf_pt_matlu $n >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgetrf_pt_matlu $n"
	./dgetrf_pt_matlu $n >> $out
	done
done
