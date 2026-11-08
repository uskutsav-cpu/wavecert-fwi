/* Devito generated code for Operator `Gradient` */

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
} ;


int Gradient(struct dataobj *restrict damp_vec, struct dataobj *restrict grad_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const double dt, const double o_x, const double o_y, const int p_rec_M, const int p_rec_m, const int time_M, const int time_m, struct profiler * timers)
{
  double (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[damp_vec->size[1]]) damp_vec->data;
  double (*restrict grad)[grad_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[grad_vec->size[1]]) grad_vec->data;
  double (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[rec_vec->size[1]]) rec_vec->data;
  double (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  double (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (double (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  double (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (double (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;
  double (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[vp_vec->size[1]]) vp_vec->data;

  const double r1 = 1.0/(dt*dt);
  const double r2 = 1.0/dt;

  for (int time = time_M, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time >= time_m; time -= 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    START(section0)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,v,vp:16)
      for (int y = y_m; y <= y_M; y += 1)
      {
        double r3 = 1.0/(vp[x + 4][y + 4]*vp[x + 4][y + 4]);
        v[t1][x + 4][y + 4] = (-r3*(-2.0*r1*v[t0][x + 4][y + 4] + r1*v[t2][x + 4][y + 4]) + r2*damp[x + 4][y + 4]*v[t0][x + 4][y + 4] + 5.33333333302289e-4*(-v[t0][x + 2][y + 4] - v[t0][x + 4][y + 2] - v[t0][x + 4][y + 6] - v[t0][x + 6][y + 4]) + 8.53333333283663e-3*(v[t0][x + 3][y + 4] + v[t0][x + 4][y + 3] + v[t0][x + 4][y + 5] + v[t0][x + 5][y + 4]) - 3.2e-2*v[t0][x + 4][y + 4])/(r3*r1 + r2*damp[x + 4][y + 4]);
      }
    }
    STOP(section0,timers)

    START(section1)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      for (int rp_recx = 0; rp_recx <= 1; rp_recx += 1)
      {
        for (int rp_recy = 0; rp_recy <= 1; rp_recy += 1)
        {
          int posx = (int)(floor(8.0e-2*(-o_x + rec_coords[p_rec][0])));
          int posy = (int)(floor(8.0e-2*(-o_y + rec_coords[p_rec][1])));
          double px = 8.0e-2*(-o_x + rec_coords[p_rec][0]) - floor(8.0e-2*(-o_x + rec_coords[p_rec][0]));
          double py = 8.0e-2*(-o_y + rec_coords[p_rec][1]) - floor(8.0e-2*(-o_y + rec_coords[p_rec][1]));
          if (rp_recx + posx >= x_m - 1 && rp_recy + posy >= y_m - 1 && rp_recx + posx <= x_M + 1 && rp_recy + posy <= y_M + 1)
          {
            double r0 = (dt*dt)*(vp[rp_recx + posx + 4][rp_recy + posy + 4]*vp[rp_recx + posx + 4][rp_recy + posy + 4])*(rp_recx*px + (1 - rp_recx)*(1 - px))*(rp_recy*py + (1 - rp_recy)*(1 - py))*rec[time][p_rec];
            v[t1][rp_recx + posx + 4][rp_recy + posy + 4] += r0;
          }
        }
      }
    }
    STOP(section1,timers)

    START(section2)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(grad,u,v:16)
      for (int y = y_m; y <= y_M; y += 1)
      {
        grad[x + 1][y + 1] += -(-2.0*r1*v[t0][x + 4][y + 4] + r1*v[t1][x + 4][y + 4] + r1*v[t2][x + 4][y + 4])*u[time][x + 4][y + 4];
      }
    }
    STOP(section2,timers)
  }

  return 0;
}
