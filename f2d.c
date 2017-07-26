/* fourier approximation to h functions. for use in computing existence of traveing wave solutions */

#include <stdio.h>
#include <math.h>
#define pi 3.1415926

/* #define x in[0] */
/* #define y in[1] */

double h(double x,double y)
/* near-complete fourier approximation to H function (i.e. no truncation) */
{
  int n[] = { 1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,
	      2,  3, -3, -2, -1};
  int m[] = { 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2, -2, -2, -2, -2, -2, -2, -1,
	      -1, -1, -1, -1, -1};
  
  double coeff_im[] = {   2.99041641e-01,  -1.23427222e-02,   2.92404663e-07,  -2.92404663e-07,
			  1.23427222e-02,  -2.99041641e-01,  -1.10662060e-01 ,  2.55677958e-03,
			  -1.30119170e-07,   1.30119170e-07,  -2.55677958e-03,   1.10662060e-01,
			  1.34078963e-03,  -8.78193376e-06  , 1.40550933e-07,  -1.40550933e-07,
			  8.78193376e-06,  -1.34078963e-03 ,  1.34078963e-03,  -8.78193376e-06,
			  1.40550933e-07,  -1.40550933e-07,   8.78193376e-06,  -1.34078963e-03,
			  -1.10662060e-01,   2.55677958e-03,  -1.30119170e-07 ,  1.30119170e-07,
			  -2.55677958e-03,   1.10662060e-01};

  double h1=0;

  int i;

  for(i=0;i<30;i++){
    h1 += -coeff_im[i]*sin((x+pi)*n[i] + (y+pi)*m[i]);    
  }

  return h1;

}

double G(double nu1,double nu2)
/* function G = int_0^\infty exp(-s)H(nu1 s, nu2 s) ds */
{
  int j;
  double g1=0;
  double ds=.1;

  for(j=0;j<500;j++){
    g1 += exp(-j*ds)*h(nu1*j*ds,nu2*j*ds);
  }
  double tfinal=500*ds;

  return g1*ds;
}

getg(double *in,double *out,int nin,int nout, double *var, double *con)
{

  float nu1=in[0];
  float nu2=in[1];

  out[0] = G(nu1,nu2);
  out[1] = G(nu2,nu1);
  
}

