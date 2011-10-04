

#define max(a,b)	( ((a)>(b)) ? (a) : (b)  )

#if defined(CONFIG_USE_FLOAT)

typedef float double_type;

#define larnv		LAPACKE_slarnv
#define lagsy		slagsy
#define lamch		LAPACKE_slamch
#define lacpy		LAPACKE_slacpy
#define cblas_trmm	cblas_strmm
#define lange		LAPACKE_slange_work
#define cblas_gemm	cblas_sgemm
#define cblas_axpy	cblas_saxpy

#define cublasGemm	cublasSgemm

#define laswp		LAPACKE_slaswp

#elif defined(CONFIG_USE_DOUBLE)

typedef double double_type;

#define larnv		LAPACKE_dlarnv
#define lagsy		dlagsy
#define lamch		LAPACKE_dlamch
#define lacpy		LAPACKE_dlacpy
#define cblas_trmm	cblas_dtrmm
#define lange		LAPACKE_dlange_work
#define cblas_gemm	cblas_dgemm
#define cblas_axpy	cblas_daxpy

#define cublasGemm	cublasDgemm

#define laswp		LAPACKE_dlaswp

#endif

