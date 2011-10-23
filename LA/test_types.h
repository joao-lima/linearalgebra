

#define max(a,b)	( ((a)>(b)) ? (a) : (b)  )

#define FMULS_POTRF(n) ((n) * (((1. / 6.) * (n) + 0.5) * (n) + (1. / 3.)))
#define FADDS_POTRF(n) ((n) * (((1. / 6.) * (n)      ) * (n) - (1. / 6.)))

#define FMULS_GETRF(n) (0.5 * (n) * ((n) * ((n) - (1./3.) * (n) - 1. ) + (n)) + (2. / 3.) * (n)) 
#define FADDS_GETRF(n) (0.5 * (n) * ((n) * ((n) - (1./3.) * (n) ) - (n)) + (1. / 6.) * (n)) 

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

#endif

