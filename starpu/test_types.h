
#ifndef __TEST_TYPES__
#define __TEST_TYPES__

#define max(a,b)	( ((a)>(b)) ? (a) : (b)  )

#if defined(CONFIG_USE_FLOAT)

typedef float double_type;

#define larnv		LAPACKE_slarnv
#define lamch		LAPACKE_slamch
#define lacpy		LAPACKE_slacpy
#define lange		LAPACKE_slange_work
#define cblas_gemm	cblas_sgemm
#define laswp		LAPACKE_slaswp
#define cblas_axpy	cblas_saxpy
#define clapack_potrf	clapack_spotrf
#define clapack_getrf	clapack_sgetrf

#define cublasGemm	cublasSgemm
#define cublasTrsm	cublasStrsm
#define cublasScal	cublasSscal
#define cublasSyr	cublasSsyr

#define GEMM		SGEMM
#define	TRSM		STRSM
#define SCAL		SSCAL
#define SYR		SSYR

#elif defined(CONFIG_USE_DOUBLE)

typedef double double_type;

#define larnv		LAPACKE_dlarnv
#define lamch		LAPACKE_dlamch
#define lacpy		LAPACKE_dlacpy
#define cblas_trmm	cblas_dtrmm
#define lange		LAPACKE_dlange_work
#define cblas_gemm	cblas_dgemm
#define laswp		LAPACKE_dlaswp
#define cblas_axpy	cblas_daxpy
#define clapack_potrf	clapack_dpotrf
#define clapack_getrf	clapack_dgetrf

#define cublasGemm	cublasDgemm
#define cublasTrsm	cublasDtrsm
#define cublasScal	cublasDscal
#define cublasSyr	cublasDsyr

#define GEMM		DGEMM
#define	TRSM		DTRSM
#define SCAL		DSCAL
#define SYR		DSYR

#endif

#endif
