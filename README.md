# cg_benchmark
Conjugate gradient related benchmark code.

Each file corresponds to a specific implementation of conjugate-gradient method:
- `taichi_cg.py` is a manual conjugate-gradient solver written in Taichi
- `eigen_cg.py` is a conjugate-gradient solver in Eigen, packaged in the `ti.linalg` module. Be aware that this is a new feature implemented in PR#7035 - you need to make sure it's included in your Taichi version before running this script. 

- `eigen_api_cg.py` is a manual conjugate-graident solver built based on basic matrix-vector operators implemented in Eigen

For time measurement, `ti.profiler` is used in `taichi_cg.py`. Sample output:

```
[Taichi] version 1.4.0, llvm 15.0.4, commit 2ba9523a, linux, python 3.9.12
[Taichi] Starting on arch=x64
>>> Solving using CGSolver...
>>> Conjugate Gradient method converged.
>>> #iterations 154
=========================================================================
Kernel Profiler(count, default) @ X64 
=========================================================================
[      %     total   count |      min       avg       max   ] Kernel name
-------------------------------------------------------------------------
[ 18.08%   0.004 s    156x |    0.015     0.026     0.946 ms] reduce_c78_0_kernel_2_range_for
[ 17.85%   0.004 s    155x |    0.017     0.026     0.412 ms] update_x_c84_0_kernel_0_range_for
[ 16.96%   0.004 s    155x |    0.016     0.025     0.442 ms] update_p_c88_0_kernel_0_range_for
[ 16.92%   0.004 s    155x |    0.020     0.025     0.094 ms] compute_Ap_c80_0_kernel_0_range_for
[ 15.30%   0.003 s    155x |    0.017     0.022     0.163 ms] update_r_c86_0_kernel_0_range_for
[ 13.68%   0.003 s    155x |    0.015     0.020     0.084 ms] reduce_c78_1_kernel_2_range_for
[  1.05%   0.000 s      1x |    0.236     0.236     0.236 ms] init_c76_0_kernel_0_serial
[  0.04%   0.000 s    155x |    0.000     0.000     0.001 ms] snode_writer_2_kernel_0_serial
[  0.04%   0.000 s    155x |    0.000     0.000     0.001 ms] reduce_c78_1_kernel_1_serial
[  0.03%   0.000 s    155x |    0.000     0.000     0.001 ms] reduce_c78_1_kernel_0_serial
[  0.03%   0.000 s    154x |    0.000     0.000     0.001 ms] snode_writer_4_kernel_0_serial
[  0.02%   0.000 s    156x |    0.000     0.000     0.002 ms] reduce_c78_0_kernel_0_serial
[  0.01%   0.000 s    156x |    0.000     0.000     0.001 ms] reduce_c78_0_kernel_1_serial
-------------------------------------------------------------------------
[100.00%] Total execution time:   0.023 s   number of results: 13
=========================================================================
>>> Time spent using CGSolver: 0.02257061004638672 sec
>>> Residual CGSolver: 9.67e-07
```

Otherwise, we use Python's `time` module to measure time collapsed for `eigen_cg.py` and `eigen_api_cg.py`. Sample output from `eigen_cg.py`:
```
[Taichi] version 1.4.0, llvm 15.0.4, commit 2ba9523a, linux, python 3.9.12
[Taichi] Starting on arch=x64
#iterations:     128
estimated error: 8.26475e-07
Time collapse: 0.005691051483154297 sec

```
and from `eigen_api_cg.py` :
```
[Taichi] version 1.4.0, llvm 15.0.4, commit 2ba9523a, linux, python 3.9.12
[Taichi] Starting on arch=x64
>>> Solving using CGSparseSolver...
>>> Conjugate Gradient method converged.
>>> #iterations 155
>>> Time spent using CGSparseSolver: 0.17 sec
```
