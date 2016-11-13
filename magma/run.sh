#!/bin/bash

#verif="1"
niter="30"
version=$(date +%s)
SCRATCH=$HOME
#SCRATCH="/scratch/jvlima"

function run_gemm {
	local prog="$1"
	local out="$2"
	local ninputs="$3"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		    echo "MAGMA_NUM_GPUS=$MAGMA_NUM_GPUS $prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dgemm {
    local out="$SCRATCH/res/magma-dgemm-${MAGMA_NUM_GPUS}gpu-$version.txt"
    local exe=" ./dgemm_matprod "
    local ninputs="256 512 $(seq 1024 1024 10240)"
    run_gemm "$exe" "$out" "$ninputs"
}

function run_dgemm_nocopy {
    local out="$SCRATCH/res/magma-dgemm-nocopy-${MAGMA_NUM_GPUS}gpu-$version.txt"
    local exe=" ./dgemm_matprod_nocopy "
    local ninputs="256 512 $(seq 1024 1024 10240)"
    run_gemm "$exe" "$out" "$ninputs"
}

function run_potrf {
	local prog="$1"
	local out="$2"
	local ninputs="$3"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		    echo "MAGMA_NUM_GPUS=$MAGMA_NUM_GPUS $prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dpotrf {
    local out="$SCRATCH/res/magma-dpotrf-${MAGMA_NUM_GPUS}gpu-$version.txt"
    local exe=" ./dpotrf_matcholesky "
    local ninputs="18432 $(seq 28672 1024 30720)"
    run_potrf "$exe" "$out" "$ninputs"
}

function run_getrf {
	local prog="$1"
	local out="$2"
	local ninputs="$3"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		    echo "MAGMA_NUM_GPUS=$MAGMA_NUM_GPUS $prog $n $out"
#		    $prog $n >> $out
		done
	done
}

function run_dgetrf {
    local prog=" ./dgetrf_matlu "
    local out="$SCRATCH/res/magma-dgetrf-$version.txt"
    local ninputs="$(seq 1024 1024 18432)"
    run_getrf "$prog" "$out" "$ninputs"
}

function run_geqrf {
	local prog="$1"
	local out="$2"
	local ninputs="$3"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		    echo "MAGMA_NUM_GPUS=$MAGMA_NUM_GPUS $prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dgeqrf {
    local prog=" ./dgeqrf_matqr"
    local out="$SCRATCH/res/magma-dgeqrf-$version.txt"
    local ninputs="256 512 $(seq 1024 1024 30720)"
    run_geqrf "$prog" "$out" "$ninputs"
}

unset MAGMA_NUM_GPUS

#run_dgemm
run_dgeqrf

exit 0

