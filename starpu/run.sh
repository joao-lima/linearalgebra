#!/bin/bash

#SCRATCH=$HOME
#SCRATCH="/scratch/jvlima"

#sched="heft"
#sched="eager"
#sched="help"
#sched="dmda"
#sched="pheft"
#sched="random"

version="$(date +%s)"

export LD_LIBRARY_PATH="$SCRATCH/install/starpu-1.0.1/lib:$LD_LIBRARY_PATH"

export STARPU_WORKERS_CPUID="9 11" 
export STARPU_WORKERS_CUDAID="7" 
#export STARPU_WORKERS_CUDAID="0 3 4 5 6 7" 


function run_potrf {
	ninputs="$(seq 2048 2048 20480)"
	ncpus=1
	ngpus=1
	nscheds="heft"
	nblocks="1024 512"
	niter=30
#	verif=1
	#nscheds=" heft eager dmda pheft random "
	for sched in $nscheds
	do
	for nb in $nblocks
	do
	for n in $ninputs
	do
	for cpu in $ncpus
	do
	for gpu in $ngpus
	do
	    out="$SCRATCH/res/starpu-dpotrf-${sched}-${cpu}cpu${gpu}gpu-$version.txt"
		for i in `seq 1 $niter`
		do
		echo "($cpu,$gpu,$sched) ./dpotrf_matcholesky $n $nb $verif >> $out"
		STARPU_SCHED=$sched STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu \
		    ./dpotrf_matcholesky $n $nb $verif >> $out
		done
	done
	done
	done
	done
    done
}

function run_getrf {
	out="$SCRATCH/res/starpu-getrf-$version.txt"
	ninputs="$(seq 64 64 1024) $(seq 256 256 18432)"
	for n in $ninputs
	do
	for cpu in $ncpus
	do
	for gpu in $ngpus
	do
		for i in `seq 1 $niter`
		do
		echo "($cpu,$gpu) ./sgetrf_matlu  -size $n -iter 1"
		STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu \
		./sgetrf_matlu -size $n -iter 1 >> $out
		done
	done
	done
	done
	ninputs="$(seq 64 64 1024) $(seq 256 256 10496)"
	for n in $ninputs
	do
	for cpu in $ncpus
	do
	for gpu in $ngpus
	do
		echo "($cpu,$gpu) ./dgetrf_matlu  -size $n -iter 1"
		for i in `seq 1 $niter`
		do
		STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu \
		./dgetrf_matlu -size $n -iter 1 >> $out
		done
	done
	done
	done
}

function run_gemm {
    msizes="$(seq 2048 2048 30720)"
    ncpus=1
    ngpus=1
    bsize="1024"
#    verif="1"
    #nscheds=" heft eager dmda pheft random "
    nscheds="heft"
    niter=30
    for sched in $nscheds
    do
    for n in $msizes
    do
    for cpu in $ncpus
    do
    for gpu in $ngpus
    do
	    out="$HOME/res/starpu-${sched}-dgemm-${cpu}cpu${gpu}gpu-${version}.txt"
	    let ' nb = n / bsize'
	    for i in `seq 1 $niter`
	    do
	    echo "(cpu=$cpu,gpu=$gpu,$sched) ./dgemm_matprod $n $nb $verif >> $out"
	    STARPU_SCHED=$sched STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu \
		./dgemm_matprod $n $nb $verif >> $out
	    done
    done
    done
    done
    done
}

function run_gemm_2 {
    msizes="$(seq 4096 2048 30720)"
    ncpus=1
    ngpus=1
    bsize="2048"
#    verif="1"
    #nscheds=" heft eager dmda pheft random "
    nscheds="heft"
    niter=30
    for sched in $nscheds
    do
    for n in $msizes
    do
    for cpu in $ncpus
    do
    for gpu in $ngpus
    do
	    out="$HOME/res/starpu-${sched}-dgemm-${cpu}cpu${gpu}gpu-${version}.txt"
	    let ' nb = n / bsize'
	    for i in `seq 1 $niter`
	    do
	    echo "(cpu=$cpu,gpu=$gpu,$sched) ./dgemm_matprod $n $nb $verif >> $out"
	    STARPU_SCHED=$sched STARPU_NCPUS=$cpu STARPU_NCUDA=$gpu \
		./dgemm_matprod $n $nb $verif >> $out
	    done
    done
    done
    done
    done
}

run_potrf
exit 0

# matlu
#run_getrf

function gemm {
    run_gemm
    run_gemm_2
}

gemm
