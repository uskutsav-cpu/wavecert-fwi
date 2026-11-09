/* Devito generated code for Operator `initdamp` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"

struct dataobj
{
  void *restrict data;
  int * size;
  unsigned long nbytes;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
  double section3;
  double section4;
  double section5;
  double section6;
  double section7;
} ;


int initdamp(struct dataobj *restrict damp_vec, const double h_x, const double h_y, const int x_M, const int x_ltkn0, const int x_m, const int x_rtkn1, const int y_M, const int y_ltkn0, const int y_m, const int y_rtkn1, const int x_size, const int y_size, struct profiler * timers)
{
  double *restrict r0_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r0_vec),64,sizeof(double)*(long)x_size);
  double *restrict r1_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r1_vec),64,sizeof(double)*(long)x_size);
  double *restrict r2_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r2_vec),64,sizeof(double)*(long)y_size);
  double *restrict r3_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r3_vec),64,sizeof(double)*(long)y_size);

  double (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[damp_vec->size[1]]) damp_vec->data;
  double *restrict r0 __attribute__ ((aligned (64))) = (double (*)) r0_vec;
  double *restrict r1 __attribute__ ((aligned (64))) = (double (*)) r1_vec;
  double *restrict r2 __attribute__ ((aligned (64))) = (double (*)) r2_vec;
  double *restrict r3 __attribute__ ((aligned (64))) = (double (*)) r3_vec;

  START(section0)
  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(damp:16)
    for (int y = y_m; y <= y_M; y += 1)
    {
      damp[x + 4][y + 4] = 0.0;
    }
  }
  STOP(section0,timers)

  START(section1)
  for (int x = x_m; x <= x_m + x_ltkn0 - 1; x += 1)
  {
    r0[x] = sin(6.283185307179590*fabs(2.5e-2*(-x + x_m) + 1.0250));
  }
  STOP(section1,timers)

  START(section2)
  for (int x = x_m; x <= x_m + x_ltkn0 - 1; x += 1)
  {
    #pragma omp simd aligned(damp:16)
    for (int y = y_m; y <= y_M; y += 1)
    {
      damp[x + 4][y + 4] += (2.5904082296183e-1*fabs(2.5e-2*(-x + x_m) + 1.0250) - 4.12276274369678e-2*r0[x])/h_x;
    }
  }
  STOP(section2,timers)

  START(section3)
  for (int x = x_M - x_rtkn1 + 1; x <= x_M; x += 1)
  {
    r1[x] = sin(6.283185307179590*fabs(2.5e-2*(x - x_M) + 1.0250));
  }
  STOP(section3,timers)

  START(section4)
  for (int x = x_M - x_rtkn1 + 1; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(damp:16)
    for (int y = y_m; y <= y_M; y += 1)
    {
      damp[x + 4][y + 4] += (2.5904082296183e-1*fabs(2.5e-2*(x - x_M) + 1.0250) - 4.12276274369678e-2*r1[x])/h_x;
    }
  }
  STOP(section4,timers)

  START(section5)
  for (int y = y_m; y <= y_m + y_ltkn0 - 1; y += 1)
  {
    r2[y] = sin(6.283185307179590*fabs(2.5e-2*(-y + y_m) + 1.0250));
  }
  STOP(section5,timers)

  START(section6)
  for (int y = y_M - y_rtkn1 + 1; y <= y_M; y += 1)
  {
    r3[y] = sin(6.283185307179590*fabs(2.5e-2*(y - y_M) + 1.0250));
  }
  STOP(section6,timers)

  START(section7)
  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(damp:16)
    for (int y = y_m; y <= y_m + y_ltkn0 - 1; y += 1)
    {
      damp[x + 4][y + 4] += (2.5904082296183e-1*fabs(2.5e-2*(-y + y_m) + 1.0250) - 4.12276274369678e-2*r2[y])/h_y;
    }
    #pragma omp simd aligned(damp:16)
    for (int y = y_M - y_rtkn1 + 1; y <= y_M; y += 1)
    {
      damp[x + 4][y + 4] += (2.5904082296183e-1*fabs(2.5e-2*(y - y_M) + 1.0250) - 4.12276274369678e-2*r3[y])/h_y;
    }
  }
  STOP(section7,timers)

  free(r0_vec);
  free(r1_vec);
  free(r2_vec);
  free(r3_vec);

  return 0;
}
