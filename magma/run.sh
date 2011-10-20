#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
#verif="1"
niter="30"
#niter="1"
version=$(date +%s)
ninputs="$(seq 256 256 10240)"
SCRATCH=$HOME

# FLOAT
out="$SCRATCH/res/magma-potrf-$version.txt"
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./spotrf_matcholesky $n $verif"
#	./spotrf_matcholesky $n $verif >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dpotrf_matcholesky $n $verif"
#	./dpotrf_matcholesky $n $verif >> $out
	done
done

out="$SCRATCH/res/magma-getrf-$version.txt"
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgetrf_matlu $n $verif"
#	./sgetrf_matlu $n $verif >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgetrf_matlu $n $verif"
#	./dgetrf_matlu $n $verif >> $out
	done
done

exit 0
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgemm_matprod $n $verif"
	./sgemm_matprod $n $verif >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./sgemm_matprod_fermi $n $verif"
	./sgemm_matprod_fermi $n $verif >> $out
	done
done


# DOUBLE
ninputs="$(seq 64 64 7168)"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgemm_matprod $n $verif"
	./dgemm_matprod $n $verif >> $out
	done
done
for n in $ninputs
do
	for i in `seq 1 $niter`
	do
	echo "./dgemm_matprod_fermi $n $verif"
	./dgemm_matprod_fermi $n $verif >> $out
	done
done
