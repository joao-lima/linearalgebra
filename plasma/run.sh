#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="1024"
ninputs="2048"
#ncores="4 8"
ncores="4"
#verif="1"
niter="1"

for n in $ninputs
do
	for p in $ncores
	do
		for i in `seq 1 $niter`
		do
		./spotrf_matcholesky $n $p $verif
		./sgetrf_matlu $n $p $verif
		./sgemm_matprod $n $p $verif
		done
	done
done
