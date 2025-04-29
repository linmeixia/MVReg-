/**********************************************************************
*  [y,B] = mexProxMonotonic(b,w)
* 
* y = armin{ sum(w.*(y-b).^2) | y(1)>=y(2)>= ....>=y(n) }
* B(k,1) = k-th distinct value in y
* B(k,2) = total weight for the k-th distinct value
*
* mex -O  -largeArrayDims  mexProxMonotonic.c
* Copyright (c) 2016 by
* Defeng Sun, Kim-Chuan Toh
*********************************************************************/
#include <mex.h>
#include <math.h>
#include <matrix.h> 

#ifndef MWSIZE_MAX
    #define  mwIndex        int
    #define  mwSignedIndex  int
    #define  mwSize         int
#endif

/*************************************************************
*   PROCEDURE mexFunction - Entry for Matlab
**************************************************************/
 void mexFunction(const int nlhs, mxArray *plhs[],
                  const int nrhs, const mxArray *prhs[])
{
   mwIndex *irb, *jcb; 
   int     n, j, k, kend, cnt;   
   double  *b, *y, *w, *btmp, *summ, *index, *B;
   double  tmpsumm; 

   if (nrhs > 2) {
      mexErrMsgTxt("mexProxMonotonic requires at most 2 input argument."); }
   if (nlhs > 2) {
      mexErrMsgTxt("mexProxMonotonic generates at most 2 output argument."); }
 
   n = mxGetM(prhs[0]); 
   if (mxGetN(prhs[0])!= 1) {
      mexErrMsgTxt("mexProxMonotonic: b should be a column vector."); }   
   if (mxIsSparse(prhs[0])) {
      btmp = mxGetPr(prhs[0]);
      irb = mxGetIr(prhs[0]); jcb = mxGetJc(prhs[0]); 
      b = mxCalloc(n,sizeof(double));       
      kend = jcb[1]; 
      for (k=0; k<kend; k++) { b[irb[k]] = btmp[k]; } 
   } else {
      b = mxGetPr(prhs[0]); 
   }   
   if (nrhs==2) {
      if (mxGetM(prhs[1]) != n) {
          mexErrMsgTxt("mexProxMonotonic: size of b and w mismatch."); }
      if (mxIsSparse(prhs[1])) {
         mexErrMsgTxt("mexProxMonotonic: w cannot be sparse."); }
      w = mxGetPr(prhs[1]);
   } else if (nrhs==1) {
      w = mxCalloc(n,sizeof(double));
      for (j=0; j<n; j++) { w[j]=1; }
   }
   /************************************************/
   plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);  
   y = mxGetPr(plhs[0]);  
   summ = mxCalloc(n,sizeof(double));
   index = mxCalloc(n,sizeof(double));   
   /************************************************/
   index[0]=0; 
   y[0] = b[0]; summ[0] = w[0];
   cnt=0; 
   for (j=1; j<n; j++) {
       cnt += 1; 
       index[cnt] = j; 
       y[cnt] = b[j]; summ[cnt] = w[j]; 
       while ((cnt>=1) && (y[cnt-1]<=y[cnt])) {
           tmpsumm = summ[cnt-1]+summ[cnt];
           y[cnt-1] = (summ[cnt-1]*y[cnt-1]+summ[cnt]*y[cnt])/tmpsumm;
           summ[cnt-1] = tmpsumm;
           cnt -= 1;
       }
   }   
   plhs[1] = mxCreateDoubleMatrix(cnt+1,2,mxREAL);
   B = mxGetPr(plhs[1]);
   for (j=0; j<=cnt; j++) {
       B[j] = y[j]; 
       B[j+cnt+1] = summ[j];
   }      
   while (n >= 0) { 
	  for (j=index[cnt]; j<=n; j++) {
	  	  y[j] = y[cnt];
      }
   	  n = index[cnt]-1;
	  cnt -= 1;
   }      
   return;   
}
/*************************************************************/
