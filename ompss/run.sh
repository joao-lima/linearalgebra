#!/bin/bash

NX_SCHEDULE="affinity"
NX_ARGS="--no-verbose"
#NX_ARGS="--disable-cuda --no-verbose"
NX_GPUS=1

version="$(date +%s)"

function run_dgemm  {
    msizes="$(seq 2048 2048 20480)"
    bsizes="1024"
    niter=30
#   verif=1
    export NX_GPUCUBLASINIT=yes
    export NX_GPUOVERLAP=yes
    export NX_GPUOVERLAP_INPUTS=yes
    export NX_GPUOVERLAP_OUTPUTS=yes
    export NX_GPUPREFETCH=yes
    export OMP_NUM_THREADS=1
    out="$HOME/res/ompss-dgemm-${NX_SCHEDULE}-${OMP_NUM_THREADS}cpu${NX_GPUS}gpu-${version}.txt"
    for b in $bsizes
    do
	for m in $msizes
	do
	    for i in `seq 1 $niter`
	    do
		echo "(ngpu=$NX_GPUS) ./dgemm_matprod $m $b $verif >> $out"
		NX_GPUS=$NX_GPUS \
		NX_SCHEDULE=$NX_SCHEDULE NX_ARGS=$NX_ARGS \
		    ./dgemm_matprod $m $b $verif >> $out
	    done
	done
    done
}

function run_dpotrf {
    msizes="$(seq 2048 2048 20480)"
    bsizes="1024"
    niter=30
#    verif=1
    export NX_GPUCUBLASINIT=yes
    export NX_GPUOVERLAP=yes
    export NX_GPUOVERLAP_INPUTS=yes
    export NX_GPUOVERLAP_OUTPUTS=yes
    export NX_GPUPREFETCH=yes
    export OMP_NUM_THREADS=1
    out="$HOME/res/ompss-dpotrf-${NX_SCHEDULE}-${OMP_NUM_THREADS}cpu${NX_GPUS}gpu-${version}.txt"
    for b in $bsizes
    do
	for m in $msizes
	do
	    for i in `seq 1 $niter`
	    do
		echo "(ngpu=$NX_GPUS) ./dpotrf_matcholesky $m $b $verif >> $out"
		NX_GPUS=$NX_GPUS \
		NX_SCHEDULE=$NX_SCHEDULE NX_ARGS=$NX_ARGS \
		    ./dpotrf_matcholesky $m $b $verif >> $out
	    done
	done
    done
}

run_dgemm
run_dpotrf

