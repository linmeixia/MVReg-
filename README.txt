MVReg+

Authors: Meixia Lin, Ziyang Zeng, Yangjing Zhang

Welcome to MVReg+, a MATLAB software package for solving matrix regression problems with nuclear norm and various Lasso-type regularizations. This package is based on the Proximal Point Algorithm with the dual Newton method for its subproblems (PPDNA).

Main Solver: ppdna_linear_nuclear_vec.m

Demonstration Scripts:
- test_synthetic.m: For matrix regression problems on 2D-shaped or randomly generated data.
- test_COVID.m: For matrix regression problems on the COVID-19 dataset.

Additional MATLAB Toolboxes for Demonstrations:
To compare our solver with Nesterov algorithm (function matrix_sparsereg in test_COVID.m), you need to install the following additional MATLAB toolboxes:
- SparseReg and TensorReg: Available at https://hua-zhou.github.io/software.html
- Tensor Toolbox (v3.5): Available at https://www.tensortoolbox.org/

The 2D-shaped matrix coefficient B is generated using the array_resize function from TensorReg.


References for Lasso-type Regularizers:
The package supports multiple Lasso-type regularizations, as referenced below:
- Lasso: X. Li, D. F. Sun, and K.-C. Toh, "A highly efficient semismooth Newton augmented Lagrangian method for solving Lasso problems". SIAM Journal on Optimization, 28.1 (2018), pp. 433–458. Code available at https://github.com/MatOpt/SuiteLasso
- Fused Lasso: X. Li, D. F. Sun, and K.-C. Toh, "On efficiently solving the subproblems of a level-set method for fused Lasso problems". SIAM Journal on Optimization, 28.2 (2018), pp. 1842–1866. Code available at https://github.com/MatOpt/SuiteLasso
- Sparse group Lasso: Y. Zhang, N. Zhang, D. F. Sun, and K.-C. Toh, "An efficient Hessian based algorithm for solving large-scale sparse group Lasso problems". Mathematical Programming, 179 (2020), pp. 223–263. Code available at https://github.com/YangjingZhang/SparseGroupLasso
- SLOPE: Z. Luo, D. F. Sun, K.-C. Toh, and N. Xiu, "Solving the OSCAR and SLOPE models using a semismooth Newton-based augmented Lagrangian method". Journal of Machine Learning Research, 20.106 (2019): 1–25.
- Exclusive Lasso: M. Lin, Y. Yuan, D. F. Sun, K.-C. Toh, "A highly efficient algorithm for solving exclusive Lasso problems". Optimization Methods and Software (2023): 1–30.
- Clustered Lasso: M. Lin, Y. Liu, D. F. Sun, K.-C. Toh, "Efficient sparse semismooth Newton methods for the clustered Lasso problem". SIAM Journal on Optimization, 29.3 (2019): 2026–2052.


Version: October 2024