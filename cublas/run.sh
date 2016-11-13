#!/bin/bash

#verif="1"
niter="30"
version=$(date +%s)
#SCRATCH="/scratch/jvlima"
SCRATCH="$HOME"
#ninputs="$(seq 64 64 1024) $(seq 256 256 18432)"

function run_sgemm {
	out="$SCRATCH/res/cublas-sgemm-$version.txt"
	ninputs="$(seq 1024 1024 10240)"
	exe=" ./sgemm_matprod "
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		echo "$exe $n"
		$exe $n >> $out
		done
	done
}

function run_dgemm {
	out="$SCRATCH/res/cublas-dgemm-block-$version.txt"
	ninputs="$(seq 64 64 2048)"
	exe="./dgemm_matprod"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		echo "$exe $n"
		$exe $n >> $out
		done
	done
}

function run_dgemm_nocopy {
	out="$SCRATCH/res/cublas-dgemm-nocopy-block-$version.txt"
	ninputs="$(seq 64 64 2048)"
	exe="./dgemm_matprod_nocopy"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		echo "$exe $n"
		$exe $n >> $out
		done
	done
}

function run_dgemm_anod2h {
	out="$SCRATCH/res/cublas-dgemm-anod2h-$version.txt"
	ninputs="256 512 $(seq 1024 1024 10240)"
	exe="./dgemm_matprod_anod2h"
	for n in $ninputs
	do
		for i in `seq 1 $niter`
		do
		echo "$exe $n"
		$exe $n >> $out
		done
	done
}

run_dgemm_anod2h
#run_dgemm_nocopy
#run_dgemm
#run_sgemm

