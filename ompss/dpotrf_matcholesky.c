
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cblas.h>
#include <lapacke.h>

extern void * nanos_malloc_pinned_cuda( size_t size );
extern void nanos_free_pinned_cuda( void * address );

extern int check_factorization(int N, double *A1, double *A2, int LDA, int uplo);

static int IONE=1;
static int ISEED[4] = {0,0,0,1};   /* initial seed for sLAPACKE_dlarnv() */

double get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}

static void
generate_matrix( double** a, size_t n, size_t blocsize )
{
    size_t i, j;
    size_t ibloc, jbloc, i_idx, j_idx;
    size_t nb = n/ blocsize;

    for (i = 0; i< n; i++) {
	ibloc = i/blocsize;
	i_idx = i%blocsize;
	a[ ibloc*nb + ibloc][i_idx*blocsize+i_idx ] =
	    a[ibloc*nb + ibloc][i_idx*blocsize+i_idx] + 1.*n; 
	for (j = 0; j < i; j++) {
	  jbloc = j/blocsize;
	  j_idx = j%blocsize;
	  a[ibloc*nb+jbloc][i_idx*blocsize+j_idx] =
	      a[jbloc*nb+ibloc][j_idx*blocsize+i_idx];
	}
    }
#if 0
    for( i = 0; i < blocsize; i++ ){
	ibloc = i/blocsize;
	i_idx = i%blocsize;
	for( j = 0; j < blocsize; j++ ) {
	  jbloc = j/blocsize;
	  j_idx = j%blocsize;
	  printf( " %.2f ",
		  a[ibloc*nb+jbloc][i_idx*blocsize+j_idx]
		);
	}
	printf("\n");
    }
#endif
}

#pragma omp target device(smp)
#pragma omp task inout([blocsize][blocsize] a)
extern void dpotrf_create_task( double* const a, const size_t blocsize );

#pragma omp target device(cuda) copy_deps
#pragma omp task inout([blocsize][blocsize] c) \
    input([blocsize][blocsize] a)
extern void dsyrk_create_task( const double* const a, double* const c, const size_t blocsize);

#pragma omp target device(cuda) copy_deps
#pragma omp task inout([blocsize][blocsize] c) \
    input([blocsize][blocsize] a)
extern void dtrsm_create_task( const double* const a, double* const c, const size_t blocsize);

#pragma omp target device(cuda) copy_deps
#pragma omp task inout([blocsize][blocsize] c) \
    input([blocsize][blocsize] a) input([blocsize][blocsize] b)
extern void dgemm_create_task( const double* const a, const double* const  b,
	double* const c, const size_t blocsize );

int
main( int argc, char **argv )
{
    size_t N = 512;
    size_t blocsize = 256;
    size_t nb;
    size_t i, j, k;
    double t0, t1;
    int verif = 0;
    double **a, *Acopy;

    if( argc > 1 )
	N = atol(argv[1]);
    if( argc > 2 )
	blocsize = atol(argv[2]);
    if (argc > 3)
	verif = atoi(argv[3]);

    nb = N / blocsize;

    //a      = (double**) malloc( nb * nb * sizeof(double*));
    a      = (double**)nanos_malloc_pinned_cuda(nb * nb * sizeof(double*));
    Acopy  = NULL;
    if ((!a)){
        printf("Out of Memory \n ");
        abort();
    }
    if( verif ) {
	Acopy = (double*) malloc( N * N * sizeof(double));
    }

    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
	    a[j*nb+i] = (double*) nanos_malloc_pinned_cuda( blocsize * blocsize * sizeof(double) );
		if( NULL == a[j*nb+i] ){
		    fprintf(stdout, "ERROR cannot allocate memory\n");
		abort();
	    }
	    LAPACKE_dlarnv( IONE, ISEED, blocsize * blocsize, a[j*nb+i] );
	}
    }
    generate_matrix( a, N, blocsize );
    if( verif ){
	size_t ibloc, i_idx, jbloc, j_idx;
	for( i = 0; i < blocsize; i++ ){
	    ibloc = i/blocsize;
	    i_idx = i%blocsize;
	    for( j = 0; j < blocsize; j++ ) {
	      jbloc = j/blocsize;
	      j_idx = j%blocsize;
	      Acopy[i*N+j] = a[ibloc*nb+jbloc][i_idx*blocsize+j_idx];
	    }
	}
    }

#define A(x,y)	    (a[nb*x+y])
    size_t m, n;
    t0 = get_elapsedtime();
    for( k = 0; k < nb; k++ ){
	dpotrf_create_task( A(k,k), blocsize );

	for( m = k+1; m < nb; m++ )
	    dtrsm_create_task( A(k,k), A(k,m), blocsize );

	for( m = k+1; m < nb; m++ ){
	    dsyrk_create_task( A(k,m), A(m,m), blocsize );

	    for( n = k+1; n < m; n++ ){
		dgemm_create_task( A(k,m), A(k,n), A(n,m), blocsize );
	    }
	}
    }
#pragma omp taskwait

    t1 = get_elapsedtime();
    double tdelta = t1 - t0;

#define FMULS_POTRF(n) ((n) * (((1. / 6.) * (n) + 0.5) * (n) + (1. / 3.)))
#define FADDS_POTRF(n) ((n) * (((1. / 6.) * (n)      ) * (n) - (1. / 6.)))
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
    double gflops = 1e-9 * FLOPS(N) / tdelta;
    printf("# size   time      GFlop/s\n");
    printf("DPOTRF %lu %lu %.10f %.6f\n", N, blocsize, tdelta, gflops);
    fflush(stdout);

    if( verif ) {
	size_t ibloc, i_idx, jbloc, j_idx;
	double *aa = (double*) malloc( N * N * sizeof(double) );
	for( i = 0; i < blocsize; i++ ){
	    ibloc = i/blocsize;
	    i_idx = i%blocsize;
	    for( j = 0; j < blocsize; j++ ) {
	      jbloc = j/blocsize;
	      j_idx = j%blocsize;
	      aa[i*N+j] = a[ibloc*nb+jbloc][i_idx*blocsize+j_idx];
	    }
	}
	check_factorization( N, Acopy, aa, N, CblasLower );
	free( aa );
    }

    for( i = 0; i < nb; i++ ){
	for( j= 0; j < nb; j++ ){
	    nanos_free_pinned_cuda( a[j*nb+i] );
	}
    }
    nanos_free_pinned_cuda(a); 
    if( verif ){
	free( Acopy );
    }
    return 0;
}

