

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

#endif

