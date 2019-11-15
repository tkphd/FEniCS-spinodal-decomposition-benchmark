# Cahn-Hilliard equation
# ======================
#
# This benchmark is implemented in a single Python file,
# which contains both the variational forms and the solver.
#
# Equation and problem definition
# -------------------------------
#
# The Cahn-Hilliard equation is a parabolic equation and is typically
# used to model phase separation in binary mixtures.  It involves
# first-order time derivatives, and second- and fourth-order spatial
# derivatives.  The equation reads:
#
# Mixed form
# ^^^^^^^^^^
#
# The Cahn-Hilliard equation is a fourth-order equation, so casting it
# in a weak form would result in the presence of second-order spatial
# derivatives, and the problem could not be solved using a standard
# Lagrange finite element basis.  A solution is to rephrase the problem
# as two coupled second-order equations:
#
# .. math::
#    \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad {\rm in} \ \Omega, \\
#    \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad {\rm in} \ \Omega.
#
# The unknown fields are now :math:`c` and :math:`\mu`. The weak
# (variational) form of the problem reads: find :math:`(c, \mu) \in V
# \times V` such that
#
# .. math::
#    \int_{\Omega} \frac{\partial c}{\partial t} q \, {\rm d} x + \int_{\Omega} M \nabla\mu \cdot \nabla q \, {\rm d} x
#           &= 0 \quad \forall \ q \in V,  \\
#    \int_{\Omega} \mu v \, {\rm d} x - \int_{\Omega} \frac{d f}{d c} v \, {\rm d} x - \int_{\Omega} \lambda \nabla c \cdot \nabla v \, {\rm d} x
#           &= 0 \quad \forall \ v \in V.
#
#
# Time discretisation
# ^^^^^^^^^^^^^^^^^^^
#
# Before being able to solve this problem, the time derivative must be
# dealt with. Apply the :math:`\theta`-method to the mixed weak form of
# the equation:
#
# .. math::
#
#    \int_{\Omega} \frac{c_{n+1} - c_{n}}{dt} q \, {\rm d} x + \int_{\Omega} M \nabla \mu_{n+\theta} \cdot \nabla q \, {\rm d} x
#           &= 0 \quad \forall \ q \in V  \\
#    \int_{\Omega} \mu_{n+1} v  \, {\rm d} x - \int_{\Omega} \frac{d f_{n+1}}{d c} v  \, {\rm d} x - \int_{\Omega} \lambda \nabla c_{n+1} \cdot \nabla v \, {\rm d} x
#           &= 0 \quad \forall \ v \in V
#
# where :math:`dt = t_{n+1} - t_{n}` and :math:`\mu_{n+\theta} =
# (1-\theta) \mu_{n} + \theta \mu_{n+1}`.  The task is: given
# :math:`c_{n}` and :math:`\mu_{n}`, solve the above equation to find
# :math:`c_{n+1}` and :math:`\mu_{n+1}`.
#
from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
from dolfin import *
from math import sin, cos
import matplotlib.pyplot as plt
#
# Benchmark parameters
# ^^^^^^^^^^^^^^^^^^^^
#
# The following domains, functions and time stepping parameters are used
# in this demo:
#
# * :math:`\Omega = (0, 200) \times (0, 200)` (unit square)
# * :math:`f = \rho (c - ca)^{2} (cb - c)^{2}`
# * :math:`\kappa = 2`
# * :math:`M = 5`
# * :math:`\epsilon = 0.5`
# * :math:`dt = 5 \times 10^{-6}`
# * :math:`\theta = 0.5`
# * :math:`L_{x} = L_{y} = 200`

ca = 0.3
cb = 0.7
rho = 5.0
M = 5.0
kappa = 2.0

#
# Implementation
# --------------

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


# Sub domain for Periodic boundary condition
# Cf. https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, onBoundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool(
            (near(x[0], 0) or near(x[1], 0))
            and (
                not (
                    (near(x[0], 0) and near(x[1], 1))
                    or (near(x[0], 1) and near(x[1], 0))
                )
            )
            and onBoundary
        )

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1]
        else:  # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.0


# This is a subclass of :py:class:`Expression
# <dolfin.functions.expression.Expression>`. The function ``eval`` returns
# values for a function of dimension two.  For the first component of the
# function, a randomized value is returned.  The method ``value_shape``
# declares that the :py:class:`Expression
# <dolfin.functions.expression.Expression>` is vector valued with dimension
# two.
#
# .. index::
#    single: NonlinearProblem; (in Cahn-Hilliard demo)
#
# A class which will represent the Cahn-Hilliard in an abstract from for
# use in the Newton solver is now defined. It is a subclass of
# :py:class:`NonlinearProblem <dolfin.cpp.NonlinearProblem>`. ::


class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)


# The constructor (``__init__``) stores references to the bilinear
# (``a``) and linear (``L``) forms. These will used to compute the
# Jacobian matrix and the residual vector, respectively, for use in a
# Newton solver.  The functions ``F`` and ``J`` are virtual member
# functions of :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>`. The function ``F`` computes the
# residual vector ``b``, and the function ``J`` computes the Jacobian
# matrix ``A``.
#
# .. index::
#    singe: form compiler options; (in Cahn-Hilliard demo)
#
# It is possible to pass arguments that control aspects of the generated
# code to the form compiler. The lines ::

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# tell the form to apply optimization strategies in the code generation
# phase and the use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory).
#
# A 200×200 square mesh with 97 (= 96 + 1) vertices in each direction is
# created, and on this mesh a :py:class:`FunctionSpace
# <dolfin.functions.functionspace.FunctionSpace>` ``ME`` is built using
# a pair of linear Lagrangian elements. ::

# Create mesh and build function space
Lx = Ly = 200
dt = 5.0e-6  # time step
runtime = 100000
theta = 0.5  # time stepping family; theta=0.5 -> Crank-Nicolson
ne = 192

mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), ne, ne)
pbc = PeriodicBoundary()
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1 * P1, constrained_domain=pbc)

# Trial and test functions of the space ``ME`` are now defined::

# Define trial and test functions
du = TrialFunction(ME)
q, v = TestFunctions(ME)

# .. index:: split functions
#
# For the test functions, :py:func:`TestFunctions
# <dolfin.functions.function.TestFunctions>` (note the 's' at the end)
# is used to define the scalar test functions ``q`` and ``v``. The
# :py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
# ``du`` has dimension two. Some mixed objects of the
# :py:class:`Function <dolfin.functions.function.Function>` class on
# ``ME`` are defined to represent :math:`u = (c_{n+1}, \mu_{n+1})` and
# :math:`u0 = (c_{n}, \mu_{n})`, and these are then split into
# sub-functions::

# Define functions
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c, mu = split(u)
c0, mu0 = split(u0)

# The line ``c, mu = split(u)`` permits direct access to the components
# of a mixed function. Note that ``c`` and ``mu`` are references for
# components of ``u``, and not copies.
#
# .. index::
#    single: interpolating functions; (in Cahn-Hilliard demo)
#
# Initial conditions are created by using the class defined at the
# beginning of the demo and then interpolating the initial conditions
# into a finite element space::

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# The first line creates an object of type ``InitialConditions``.  The
# following two lines make ``u`` and ``u0`` interpolants of ``u_init``
# (since ``u`` and ``u0`` are finite element functions, they may not be
# able to represent a given function exactly, but the function can be
# approximated by interpolating it in a finite element space).
#
# .. index:: automatic differentiation
#
# The chemical potential :math:`df/dc` is computed using automated
# differentiation::

# Compute the chemical potential df/dc
c = variable(c)
f = rho * (c - ca) ** 2 * (cb - c) ** 2
dfdc = diff(f, c)

# The first line declares that ``c`` is a variable that some function
# can be differentiated with respect to. The next line is the function
# :math:`f` defined in the problem statement, and the third line
# performs the differentiation of ``f`` with respect to the variable
# ``c``.
#
# It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu

# which is then used in the definition of the variational forms::

# Weak statement of the equations
L0 = c * q * dx - c0 * q * dx + dt * dot(grad(mu_mid), grad(q)) * dx
L1 = mu * v * dx - dfdc * v * dx - kappa * dot(grad(c), grad(v)) * dx
L = L0 + L1

# This is a statement of the time-discrete equations presented as part
# of the problem statement, using UFL syntax. The linear forms for the
# two equations can be summed into one form ``L``, and then the
# directional derivative of ``L`` can be computed to form the bilinear
# form which represents the Jacobian matrix::

# Compute directional derivative about u in the direction of du (Jacobian)
a = derivative(L, u, du)

# .. index::
#    single: Newton solver; (in Cahn-Hilliard demo)
#
# The DOLFIN Newton solver requires a :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>` object to solve a system of nonlinear
# equations. Here, we are using the class ``CahnHilliardEquation``,
# which was declared at the beginning of the file, and which is a
# sub-class of :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>`. We need to instantiate objects of both
# ``CahnHilliardEquation`` and :py:class:`NewtonSolver
# <dolfin.cpp.NewtonSolver>`::

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "residual" # or "incremental"
solver.parameters["relative_tolerance"] = 1e-12

# The string ``"lu"`` passed to the Newton solver indicated that an LU
# solver should be used.  The setting of
# ``parameters["convergence_criterion"] = "incremental"`` specifies that
# the Newton solver should compute a norm of the solution increment to
# check for convergence (the other possibility is to use ``"residual"``,
# or to provide a user-defined check). The tolerance for convergence is
# specified by ``parameters["relative_tolerance"] = 1e-6``.

# Create arrays and define the system free energy
timestep = []
freeEnrg = []
F = f + 0.5 * kappa * dot(grad(c), grad(c))

# To run the solver and save the output to a VTK file for later visualization,
# the solver is advanced in time from :math:`t_{n}` to :math:`t_{n+1}` until
# a terminal time :math:`T` is reached::

with open("results/free_energy_A.csv", "w") as logfile:
    logfile.write("time,free_energy,dt\n")

    # Viz file
    file = File("results/output_A.pvd", "compressed")

    # Write initial condition
    t = 0.0
    tout = 1.0
    Ftot = assemble(F * dx)
    file << (u.split()[0], t)
    timestep.append(t)
    freeEnrg.append(Ftot)
    logfile.write("{0},{1},{2}\n".format(t, Ftot, dt))
    converged = True

    # Step in time
    while t < runtime and converged:
        u0.vector()[:] = u.vector()
        nIt, converged = solver.solve(problem, u.vector())
        t += dt
        Ftot = assemble(F * dx)
        timestep.append(t)
        freeEnrg.append(Ftot)
        logfile.write("{0},{1},{2}\n".format(t, assemble(F * dx), dt))
        if t > tout:
            tout += 1.0
            file << (u.split()[0], t)

        # Ad-hoc adaptive timestep
        if nIt < 3:
            dt *= 1.0001
        else:
            nIt *= 0.8

    if not converged:
        print("Error: solver failed to converge.")

file << (u.split()[0], t)

# The string ``"compressed"`` indicates that the output data should be
# compressed to reduce the file size. Within the time stepping loop, the
# solution vector associated with ``u`` is copied to ``u0`` at the
# beginning of each time step, and the nonlinear problem is solved by
# calling :py:func:`solver.solve(problem, u.vector())
# <dolfin.cpp.NewtonSolver.solve>`, with the new solution vector
# returned in :py:func:`u.vector() <dolfin.cpp.Function.vector>`. The
# ``c`` component of the solution (the first component of ``u``) is then
# written to file at every time step.

plt.semilogy(timestep, freeEnrg)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.savefig("results/energy_A.png", dpi=400, bbox_inches="tight")
plt.close()

print("Peak memory usage was ", getrusage(RUSAGE_SELF)[2] / 1024.0, "MB")
print("Child memory usage was ", getrusage(RUSAGE_CHILDREN)[2] / 1024.0, "MB")
