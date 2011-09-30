#!/bin/bash

#ninputs="512 1024"
ninputs="4096"
#ninputs="1024"
#ninputs="8192"
#verif="1"
niter="1"

for n in $ninputs
do
	for i in `seq 1 $niter`
	do
#	./spotrf_matcholesky $n $verif
#	./sgetrf_matlu $n $verif
	./sgemm_matprod $n $verif
	done
done
