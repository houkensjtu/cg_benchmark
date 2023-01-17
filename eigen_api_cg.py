import taichi as ti
import math
import numpy as np
import time

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.data_oriented
class CGSparseSolver:
    def __init__(self, n=256, eps=1e-6, offset=0.0, quiet=False, dtype=ti.f64):
        self.N = n
        self.eps = eps
        self.real = dtype
        self.offset = offset
        self.N_ext = 0
        self.N_tot = self.N + 2 * self.N_ext
        self.steps = self.N * self.N  # Cg should converge within the size of the vector
        self.quiet = quiet
        self.history = []  # Convergence history data
        # -- Conjugate gradient variables --
        self.alpha = 0.0
        self.beta = 0.0
        self.Ka = ti.linalg.SparseMatrixBuilder(self.N*self.N, self.N*self.N, max_num_triplets=5*self.N**2, dtype=self.real)
        self.Kb = ti.linalg.SparseMatrixBuilder(self.N*self.N, 1,             max_num_triplets=self.N**2, dtype=self.real)
        self.Kr = ti.linalg.SparseMatrixBuilder(self.N*self.N, 1,             max_num_triplets=self.N**2, dtype=self.real)
        self.Kx = ti.linalg.SparseMatrixBuilder(self.N*self.N, 1,             max_num_triplets=self.N**2, dtype=self.real)
        self.Kp = ti.linalg.SparseMatrixBuilder(self.N*self.N, 1,             max_num_triplets=self.N**2, dtype=self.real)                    

    @ti.kernel
    def init(self, Ka:ti.types.sparse_matrix_builder(), Kb:ti.types.sparse_matrix_builder(),
                   Kr:ti.types.sparse_matrix_builder(), Kx:ti.types.sparse_matrix_builder(),
                   Kp:ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(self.N, self.N):
            row = i * self.N + j
            if j != 0:
                Ka[row, row - 1] += -1.0
            if j != self.N - 1:
                Ka[row, row + 1] += -1.0
            if i != 0:
                Ka[row, row - self.N] += -1.0
            if i != self.N - 1:
                Ka[row, row + self.N] += -1.0
            Ka[row, row] += 4.0
        for i, j in ti.ndrange(self.N, self.N):
            row = i * self.N + j
            xl = i / (self.N_tot - 1)
            yl = j / (self.N_tot - 1)
            Kb[row, 0] += ti.sin(2.0 * math.pi * xl) * ti.sin(2.0 * math.pi * yl)
            Kr[row, 0] += ti.sin(2.0 * math.pi * xl) * ti.sin(2.0 * math.pi * yl)
            Kx[row, 0] += 0.0
            Kp[row, 0] += 0.0


    def update_p(self):
        self.p = self.r + self.beta * self.p


    def update_x(self):
        self.x += self.alpha * self.p


    def update_r(self):
        self.r -= self.alpha * (self.A @ self.p)
        

    def solve(self):
        self.init(self.Ka, self.Kb, self.Kr, self.Kx, self.Kp)
        self.A = self.Ka.build()
        self.b = self.Kb.build()
        self.r = self.Kr.build()
        self.x = self.Kx.build()
        self.p = self.Kp.build()
        initial_rTr = (self.r.transpose() @ self.r)[0, 0]  # Compute initial residual
        if not self.quiet:
            print('Initial residual =', ti.sqrt(initial_rTr))
        old_rTr = initial_rTr
        self.update_p()  # Initial p = r + beta * p ( beta = 0 )
        # -- Main loop --
        for i in range(self.steps):
            # 1. Compute alpha
            pAp = (self.p.transpose() @ (self.A @ self.p))[0, 0]
            self.alpha = old_rTr / pAp
            # 2. Update x and r using alpha
            self.update_x()
            self.update_r()
            # 3. Check for convergence
            new_rTr = (self.r.transpose() @ self.r)[0, 0]
            if ti.sqrt(new_rTr) < self.eps:
                print('>>> Conjugate Gradient method converged.')
                print('>>> #iterations', i)
                break
            # 4. Compute beta
            self.beta = new_rTr / old_rTr
            # 5. Update p using beta
            self.update_p()
            old_rTr = new_rTr
            #self.history.append(
            #    f'{ti.sqrt(new_rTr):e}\n'
            #)  # Write converge history; i+1 because starting from 1.
            # Visualizations
            if not self.quiet:
                print(f'Iter = {i+1:4}, Residual = {ti.sqrt(new_rTr):e}')


if __name__ == '__main__':
    cgsolver = CGSparseSolver(n=128, quiet=True)
    print('>>> Solving using CGSparseSolver...')
    now = time.time()
    cgsolver.solve()
    print(f'>>> Time spent using CGSparseSolver: {time.time() - now:4.2f} sec')    
    # print(cgsolver.x)
