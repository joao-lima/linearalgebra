#!/bin/bash

#ninputs="512 1024"
#ninputs="4096"
#ninputs="8192"
#ninputs="1024"
#ninputs="2048"
#verif="1"
ncpus="$(seq 2 2 8)"
ngpus="1"
niter="30"
version=$(date +%s)
out="$HOME/res/starpu-gemm-$version.txt"
SCRATCH=$HOME
export LD_LIBRARY_PATH="$SCRATCH/install/starpu-0.9.2/lib:$LD_LIBRARY_PATH"
#export STARPU_WORKERS_CUDAID="0 3 4 5 6 7" 
export STARPU_WORKERS_CUDAID="0" 

# 256 ate 1024
# 1024 dpeois 

ninputs="$(seq 3072 768 10752)"

for n in $ninputs
do
	# blocksize 
	let "nblocks= $n / 768" 
	for cpu in $ncpus
	do
	for gpu in $ngpus
	do
	for i in `seq 1 $niter`
	do
	echo "STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu STARPU_NOPENCL=0 \
	./sgemm_matprod  -x $n -y $n -z $n -iter 1 -nblocks $nblocks"
	STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu STARPU_NOPENCL=0 \
	./sgemm_matprod  -x $n -y $n -z $n -iter 1 -nblocks $nblocks >> $out
	done
	done
	done
done

for n in $ninputs
do
	# blocksize = 768
	let "nblocks= $n / 768" 
	for cpu in $ncpus
	do
	for gpu in $ngpus
	do
	for i in `seq 1 $niter`
	do
	echo "STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu STARPU_NOPENCL=0 \
	./dgemm_matprod  -x $n -y $n -z $n -iter 1 -nblocks $nblocks"
	STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu STARPU_NOPENCL=0 \
	./dgemm_matprod  -x $n -y $n -z $n -iter 1 -nblocks $nblocks >> $out
	done
	done
	done
done

exit 0
n=3072
nblocks=4
STARPU_NCPUS=4 STARPU_NCUDA=1 STARPU_NOPENCL=0 \
	./sgemm_matprod  -x $n -y $n -z $n -iter 1 -nblocks $nblocks 
exit 0

