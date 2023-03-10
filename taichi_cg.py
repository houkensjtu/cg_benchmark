import taichi as ti
import numpy as np
import time

ti.init(arch=ti.cpu, default_fp=ti.f64, kernel_profiler=True)

@ti.data_oriented
class CGSolver:

    def __init__(self, n=256, eps=1e-6, offset=0.0, quiet=False):
        self.N = n
        self.real = ti.f64
        self.eps = eps
        self.offset = offset
        self.N_tot = self.N + 2
        self.N_ext = 1
        self.steps = self.N * self.N  # Cg should converge within the size of the vector
        self.quiet = quiet
        self.history = []  # Convergence history data
        # -- Conjugate gradient variables --
        self.r = ti.field(dtype=self.real)  # residual
        self.b = ti.field(dtype=self.real)  # residual
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.Ax = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        ti.root.place(self.alpha, self.beta)  # , self.sum)
        ti.root.dense(ti.ij, (self.N_tot, self.N_tot)).place(self.x, self.p, self.Ap, self.r, self.Ax, self.b) # Dense data structure

    @ti.kernel
    def init(self):
        ti.loop_config(serialize=True)
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            # xl, yl, zl = [0,1]
            xl = (i - self.N_ext) / (self.N_tot - 3)
            yl = (j - self.N_ext) / (self.N_tot - 3)
            # r[0] = b - Ax, where x = 0; therefore r[0] = b
            self.r[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
            self.b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
            self.Ap[i, j] = 0.0
            self.Ax[i, j] = 0.0
            self.x[i, j] = 0.0            
            self.p[i, j] = 0.0

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()) -> ti.f64:
        sum = 0.0
        for I in ti.grouped(p):
            sum += p[I] * q[I]
        return sum

    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.Ap[i, j] = (4.0 + self.offset) * self.p[i, j] - self.p[
                i + 1, j] - self.p[i - 1, j] - self.p[i, j + 1] - self.p[i,
                                                                         j - 1]

    @ti.kernel
    def compute_Ax(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.Ax[i, j] = (4.0 + self.offset) * self.x[i, j] - self.x[
                i + 1, j] - self.x[i - 1, j] - self.x[i, j + 1] - self.x[i,
                                                                         j - 1]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.r[I] + self.beta[None] * self.p[I]

    def solve(self):
        self.init()
        initial_rTr = self.reduce(self.r, self.r)  # Compute initial residual
        if not self.quiet:
            print('Initial residual =', ti.sqrt(initial_rTr))
        # self.history.append(f'{ti.sqrt(initial_rTr):e}\n')
        old_rTr = initial_rTr
        self.update_p()  # Initial p = r + beta * p ( beta = 0 )
        # -- Main loop --
        for i in range(self.steps):
            # 1. Compute alpha
            self.compute_Ap()
            pAp = self.reduce(self.p, self.Ap)
            self.alpha[None] = old_rTr / pAp
            # 2. Update x and r using alpha
            self.update_x()
            self.update_r()
            # 3. Check for convergence
            new_rTr = self.reduce(self.r, self.r)
            if ti.sqrt(new_rTr) < self.eps:
                print('>>> Conjugate Gradient method converged.')
                print('>>> #iterations', i)
                break
            # 4. Compute beta
            self.beta[None] = new_rTr / old_rTr
            # 5. Update p using beta
            self.update_p()
            old_rTr = new_rTr
            #self.history.append(
            #    f'{ti.sqrt(new_rTr):e}\n'
            #)  # Write converge history; i+1 because starting from 1.
            # Visualizations
            if not self.quiet:
                print(f'Iter = {i+1:4}, Residual = {ti.sqrt(new_rTr):e}')

    def save_history(self):
        with open('convergence.txt', 'w') as f:
            for line in self.history:
                f.write(line)

    @ti.kernel
    def compute_residual(self):  # compute r = Ax - b
        for I in ti.grouped(self.b):
            self.r[I] = self.b[I] - self.Ax[I]

    def check_solution(self):  # Return the norm of rTr as the residual
        self.compute_Ax()
        self.compute_residual()
        return np.sqrt(self.reduce(self.r, self.r))


if __name__ == '__main__':
    psize = 128
    cgsolver = CGSolver(psize, 1e-6, offset=0.0, quiet=True) # quiet=True to print residual
    ti.profiler.clear_kernel_profiler_info()
    # Solve in Taichi using custom CG, A is implicitly represented in compute_Ap()
    print('>>> Solving using CGSolver...')
    cgsolver.solve()
    ti.sync()
    ti.profiler.print_kernel_profiler_info()
    total_time = ti.profiler.get_kernel_profiler_total_time()
    print(f'>>> Time spent using CGSolver: {total_time} sec')

    # Compare the residuals: norm(r) where r = Ax - b
    residual_cg = cgsolver.check_solution()
    print(f'>>> Residual CGSolver: {residual_cg:4.2e}')
