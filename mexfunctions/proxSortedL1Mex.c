#include "mex.h"
#include "proxSortedL1.h"

/*
 * Copyright 2013, M. Bogdan, E. van den Berg, W. Su, and E.J. Candes
 *
 * This file is part of SLOPE Toolbox version 1.0.
 *
 *   The SLOPE Toolbox is free software: you can redistribute it
 *   and/or  modify it under the terms of the GNU General Public License
 *   as published by the Free Software Foundation, either version 3 of
 *   the License, or (at your option) any later version.
 *
 *   The SLOPE Toolbox is distributed in the hope that it will
 *   be useful, but WITHOUT ANY WARRANTY; without even the implied
 *   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *   See the GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with the SLOPE Toolbox. If not, see
 *   <http://www.gnu.org/licenses/>.
 */

int evaluateProx(double *y, double *lambda, double *x, size_t n, int *order)
/* ----------------------------------------------------------------------- */
{  double   d;
   double  *s     = NULL;
   double  *w     = NULL;
   size_t  *idx_i = NULL;
   size_t *idx_j = NULL;
   size_t  i,j,k;
   int      result = 0;
   
   /* Allocate memory */
   s     = (double *)malloc(sizeof(double) * n);
   w     = (double *)malloc(sizeof(double) * n);
   idx_i = (size_t *)malloc(sizeof(size_t) * n);
   idx_j = (size_t *)malloc(sizeof(size_t) * n);
   
   if ((s != NULL) && (w != NULL) && (idx_i != NULL) && (idx_j != NULL))
   {
      k = 0;
      for (i = 0; i < n; i++)
      {
         idx_i[k] = i;
         idx_j[k] = i;
         s[k]     = y[i] - lambda[i];
         w[k]     = s[k];
         
         while ((k > 0) && (w[k-1] <= w[k]))
         {  k --;
            idx_j[k] = i;
            s[k]    += s[k+1];
            w[k]     = s[k] / (i - idx_i[k] + 1);
         }
         
         k++;
      }
      
      if (order == NULL)
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[i] = d;
            }
         }
      }
      else
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[order[i]] = d;
            }
         }
      }
   }
   else
   {  result = -1;
   }
   
   /* Deallocate memory */
   if (s     != NULL) free(s);
   if (w     != NULL) free(w);
   if (idx_i != NULL) free(idx_i);
   if (idx_j != NULL) free(idx_j);
   
   return result;
}
/* Input: Vectors y, lambda, both non-negative and in decreasing order. */

/* ----------------------------------------------------------------------- */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
/* ----------------------------------------------------------------------- */
{  mxArray *vectorY;
   mxArray *vectorLambda;
   mxArray *vectorX;
   mwSize   n, i, j, jMax, k;
   double  *y;
   double  *lambda;
   double  *x;
   
   /* Exit if no parameters are given */
   if (nrhs == 0)
   {  return ;
   }

   /* Check for proper number of arguments */
   if (nrhs != 2) mexErrMsgTxt("Two input arguments expected.");
   if (nlhs  > 2) mexErrMsgTxt("Too many output arguments.");

    /* Extract the arguments */
    vectorY      = (mxArray *)prhs[0];
    vectorLambda = (mxArray *)prhs[1];

    /* Both vectors should be real, double, n-by-1 */
    n = mxGetNumberOfElements(vectorY); /* For code symmetry */
    if ((!mxIsDouble(vectorY)) || (mxIsComplex(vectorY)) ||
        (mxGetM(vectorY) != n) || (mxGetN(vectorY) != 1) ||
        (mxGetNumberOfDimensions(vectorY) != 2))
    {  mexErrMsgTxt("First input argument should be a real n-by-1 vector of type double.\n");
    }
    
    if ((!mxIsDouble(vectorLambda)) || (mxIsComplex(vectorLambda)) ||
        (mxGetM(vectorLambda) != n) || (mxGetN(vectorLambda) != 1) ||
        (mxGetNumberOfDimensions(vectorLambda) != 2))
    {  mexErrMsgTxt("Second input argument should be a real n-by-1 vector of type double.\n");
    }

    
    /* --- DO NOT CHECK DECREASING ORDER AND SIGN PATTERNS --- */
    

    /* Allocate output argument */
    vectorX =  mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[0] = vectorX;

    /* Get data pointers */
    lambda = mxGetPr(vectorLambda);
    y = mxGetPr(vectorY);
    x = mxGetPr(vectorX);

    
    /* Solve prox function */
    evaluateProx(y,lambda,x,(size_t)n,NULL);
    
    return ;
}
