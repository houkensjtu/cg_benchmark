import numpy as np
import math
import time
import taichi as ti

ti.init(arch=ti.x64, offline_cache=False)

GRID = 128
SIZE = GRID ** 2

K = ti.linalg.SparseMatrixBuilder(SIZE, SIZE, max_num_triplets= SIZE*5)
b_ti = ti.field(shape=(SIZE,), dtype=ti.f64)

@ti.kernel
def fill_K(K: ti.types.sparse_matrix_builder()):
    for i,j in ti.ndrange(GRID, GRID):
        row = i * GRID + j
        if j != 0:
            K[row, row - 1] += -1.0
        if j != GRID - 1:
            K[row, row + 1] += -1.0
        if i != 0:
            K[row, row - GRID] += -1.0
        if i != GRID - 1:
            K[row, row + GRID] += -1.0
        K[row, row] += 4.0

@ti.kernel
def fill_b():
    for i, j in ti.ndrange(GRID, GRID):
        idx = i * GRID + j
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b_ti[idx] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)

fill_K(K)
A = K.build()
fill_b()
b = b_ti.to_numpy()
x0 = np.zeros(SIZE, dtype=np.float64)
cg = ti.linalg.CG(A, b, x0, max_iter=5000, atol=1e-6)

now = time.time()
x = cg.solve()
print('Time collapse:', time.time() - now, 'sec')
