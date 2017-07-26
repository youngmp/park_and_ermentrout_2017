#include <math.h>
#define pi 3.1415926
get2d(double *in,double *out,int nin,int nout, double *var, double *con)
{
  double a0=in[0],a10=in[1],a01=in[2],b10=in[3],b01=in[4],a11=in[5];
  double b11=in[6],c11=in[7],d11=in[8],r=in[9],theta=in[10];
  int n=50;
  double p0=0,p10=0,p01=0,q10=0,q01=0,p11=0,q11=0,r11=0,s11=0;
  double cx,cy,sx,sy,u,f;
  double dx=pi/n,dx2=dx*dx;

  int i,j;

  double cxf,cyf,sxf;


  /*   printf("%d %g %g %g %g %g %g \n",n,a0,a10,a01,a11,r,theta); */
  for(i=-n;i<n;i++){
      cy=cos(i*dx);
      sy=sin(i*dx);

    for(j=-n;j<n;j++){
      cx=cos(j*dx);
    
      sx=sin(j*dx);


      u=a0+a10*cx+a01*cy+a11*cx*cy+b10*sx+b01*sy+b11*sx*sy+c11*sx*cy+d11*sy*cx;
      /*u=a0+cx*(a10+a11*cy+d11*sy)+a01*cy+sx*(b10+b11*sy+c11*cy)+b01*sy;*/

      f=1/(1+exp(-r*(u-theta)));

      cxf = cx*f;
      cyf = cy*f;
      sxf = sx*f;


      p0+=f; /* a0 */
      p10+=(cxf); /* a10 */
      p01+=(cyf); /* a01 */
      p11+=(cy*cxf); /* a11 */
      q10+=(sxf); /* b10 */
      q01+=(sy*f); /* b01 */
      q11+=(sy*sxf); /* b11 */
      r11+=(sx*cyf); /* c11 */
      s11+=(sy*cxf); /* d11 */
   
    }
  }
  out[0]=p0*dx2; /* a0 */
  out[1]=p10*dx2; /* a10 */
  out[2]=p01*dx2;
  out[3]=q10*dx2;
  out[4]=q01*dx2;
  out[5]=p11*dx2;
  out[6]=q11*dx2;
  out[7]=r11*dx2;
  out[8]=s11*dx2;
 

}

	
  
  
