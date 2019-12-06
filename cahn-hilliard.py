# Cahn-Hilliard equation
# ======================
#
# This benchmark is implemented in a single Python file, which contains both the
# variational forms and the solver.

from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
from dolfin import *
from math import sin, cos
import matplotlib.pyplot as plt
from sys import exit
import numpy as np

ca = Constant(0.3)
cb = Constant(0.7)
rho = Constant(5.0)
M = Constant(5.0)
kappa = Constant(2.0)

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.5 + 0.01 * (
            cos(0.105 * x[0]) * cos(0.110 * x[1])
            + (cos(0.130 * x[0]) * cos(0.087 * x[1])) ** 2
            + cos(0.025 * x[0] - 0.15 * x[1]) * cos(0.070 * x[0] - 0.020 * x[1])
        )
        values[1] = 0.0

    def value_shape(self):
        return (2,)

class PeriodicBoundary(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS, length=1.0, length_scaling=1.0):
        SubDomain.__init__(self)
        self.tol = tolerance
        self.length = length
        self.length_scaling = length_scaling
        self.L = self.length / self.length_scaling

    def inside(self, x, onBoundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool(
            (near(x[0], 0) or near(x[1], 0))
            and (
                not (
                    (near(x[0], 0) and near(x[1], self.L))
                    or (near(x[0], self.L) and near(x[1], 0))
                )
            )
            and onBoundary
        )

    def map(self, x, y):
        if near(x[0], self.L) and near(x[1], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1] - self.L
        elif near(x[0], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1]
        else:  # near(x[1], self.L)
            y[0] = x[0]
            y[1] = x[1] - self.L

class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math"

# Create mesh and build function space
Lx = Ly = 200.0
runtime = 10
theta = Constant(0.5)  # time stepping family; theta=(0, ½, 1) -> (Forward, Crank-Nicolson, Backward)
ne = 192
deg = 1
rtol = 1e-6
adapt_steps = 10
dt = rtol / 4
debug = True

mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), ne, ne)
pbc = PeriodicBoundary(length=Lx, length_scaling=1.0)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), degree=deg)
ME = FunctionSpace(mesh, P1 * P1, constrained_domain=pbc)

# Define trial and test functions
du = TrialFunction(ME)
q, v = TestFunctions(ME)

# Define functions
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step
ub = Function(ME)  # adaptivity solution

# Split mixed functions
dc, dmu = split(du)
c, mu = split(u)
c0, mu0 = split(u0)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=deg)
u.interpolate(u_init)
u0.interpolate(u_init)

# Compute the chemical potential df/dc
c = variable(c)
f = rho * (c - ca) ** 2 * (cb - c) ** 2
dfdc = diff(f, c)
F = f + 0.5 * kappa * dot(grad(c), grad(c))

# It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::
mu_mid = (1.0 - theta) * mu0 + theta * mu

# Weak statement of the equations
L0 = c * q * dx - c0 * q * dx + dt * dot(grad(mu_mid), grad(q)) * dx
L1 = mu * v * dx - dfdc * v * dx - kappa * dot(grad(c), grad(v)) * dx
Lcn = L0 + L1
# Lcn = ( (c - c0) * q + dt * dot(grad(mu_mid), grad(q))
#      + (mu - dfdc) * v - kappa * dot(grad(c), grad(v))) * dx(mesh)

# Compute directional derivative about u in the direction of du (Jacobian)
acn = derivative(Lcn, u, du)

# Build the solver
crank_nicolson_problem = CahnHilliardEquation(acn, Lcn)
implicit_problem = CahnHilliardEquation(acn, Lcn)

solver = NewtonSolver()
solver.parameters["linear_solver"] = "gmres"
solver.parameters["preconditioner"] = "ilu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = rtol

beSolver = NewtonSolver()
beSolver.parameters["linear_solver"] = "gmres"
beSolver.parameters["preconditioner"] = "ilu"
beSolver.parameters["convergence_criterion"] = "incremental"
beSolver.parameters["relative_tolerance"] = 1.0e-3

# Sample output on logarithmic timeline
tOut = np.outer(
    np.array([[1], [10], [100], [1000], [10000], [100000]]), np.array([[2, 5, 10]])
).flatten()

dbgfile = open("results/adaptive.csv", "w")


# March in time
with open("results/free_energy_A.csv", "w") as logfile:
    logfile.write("time,free_energy\n")
    dbgfile.write("dt,ub-u,u,quot,error,dτ\n")

    # Viz file
    file = File("results/output_A.pvd", "compressed")

    # Write initial condition
    i = 0
    adapt = adapt_steps - 5
    t = 0.0
    file << (u.split()[0], t)
    Ftot = assemble(F * dx(mesh))
    logfile.write("{0},{1}\n".format(t, Ftot))
    logfile.flush()

    # Step in time
    while t < runtime:
        t += dt
        u0.vector()[:] = u.vector()
        solver.solve(crank_nicolson_problem, u.vector())

        # Log free energy
        Ftot = assemble(F * dx(mesh))
        logfile.write("{0},{1}\n".format(t, Ftot))

        if t > tOut[i]:
            # Visualize field
            file << (u.split()[0], t)
            i += 1
            logfile.flush()

        adapt += 1
        if debug or adapt == adapt_steps:
            # The Cahn-Hilliard equation takes a while to reach steady state, so adaptive
            # time-stepping is in order. Algorithm after *J. Comp. Phys.* 230 (2011) 5317.
            adapt = 0
            error = 1.0
            dt0 = dt
            ub.vector()[:] = u0.vector()
            beSolver.solve(implicit_problem, ub.vector())
            deltaU = errornorm(ub, u, norm_type="l2", mesh=mesh)
            residU = norm(u, norm_type="l2", mesh=mesh)
            quot = deltaU / residU
            error = sqrt(beSolver.parameters["relative_tolerance"] / quot)
            dt = 0.9 * dt0 * quot
            if debug:
                dbgfile.write("{0},{1},{2},{3},{4},{5}\n".format(dt0, deltaU, residU, quot, error, dt))
            if dt / 0.9 < dt0:
                exit(
                    "\nERROR: previous timestep was too large! {0:.2e} -> {1:.2e}\n".format(
                        dt0, dt
                    )
                )
            else:
                print("New timestep: ", dt)

# Write final visualization
file << (u.split()[0], t)

print("Peak memory usage was ", getrusage(RUSAGE_SELF)[2] / 1024.0, "MB")
print("Child memory usage was ", getrusage(RUSAGE_CHILDREN)[2] / 1024.0, "MB")
