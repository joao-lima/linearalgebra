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
		    echo "$prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dgemm {
    local out="$SCRATCH/res/atlas-dgemm-pt-$version.txt"
    local exe=" ./dgemm_pt_matprod"
    local ninputs="256 512 $(seq 1024 1024 30720)"
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
		    echo "$prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dpotrf {
    local out="$SCRATCH/res/atlas-dpotrf-pt-$version.txt"
    local exe=" ./dpotrf_pt_matcholesky "
    local ninputs="256 512 $(seq 1024 1024 30720)"
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
		    echo "$prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dgetrf {
    local prog=" ./dgetrf_pt_matlu "
    local out="$SCRATCH/res/atlas-dgetrf-pt-$version.txt"
    local ninputs="256 512 $(seq 1024 1024 30720)"
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
		    echo "$prog $n $out"
		    $prog $n >> $out
		done
	done
}

function run_dgeqrf {
    local prog=" ./dgeqrf_pt_matqr "
    local out="$SCRATCH/res/atlas-dgeqrf-pt-$version.txt"
    local ninputs="256 512 $(seq 1024 1024 30720)"
    run_geqrf "$prog" "$out" "$ninputs"
}

run_dgemm
run_dpotrf
run_dgetrf
run_dgeqrf

exit 0

