# Spinodal Decomposition Benchmark in FEniCS

This repository contains an implementation of the
[PFHub](https://pages.nist.gov/pfhub) Benchmark Problem 1, [Spinodal
Decomposition](https://pages.nist.gov/chimad-phase-field/benchmarks/benchmark1.ipynb/)
using [FEniCS](https://www.fenicsproject.org).

### Numerical Methods

FEniCS offers the following solvers and preconditioners, listed using
`print(list_linear_solver_methods())` and `print(list_krylov_solver_preconditioners())`,
respectively.

Solver method | Description
------------- | -----------
bicgstab      | Biconjugate gradient stabilized method
cg            | Conjugate gradient method
default       | default linear solver
gmres         | Generalized minimal residual method
minres        | Minimal residual method
mumps         | MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
petsc         | PETSc built in LU solver
richardson    | Richardson method
tfqmr         | Transpose-free quasi-minimal residual method
umfpack       | UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)

Preconditioner   | Description
---------------- | -----------
amg              |  Algebraic multigrid
default          |  default preconditioner
hypre\_amg       |  Hypre algebraic multigrid (BoomerAMG)
hypre\_euclid    |  Hypre parallel incomplete LU factorization
hypre\_parasails |  Hypre parallel sparse approximate inverse
icc              |  Incomplete Cholesky factorization
ilu              |  Incomplete LU factorization
jacobi           |  Jacobi iteration
one             |  No preconditioner
petsc\_amg       |  PETSc algebraic multigrid
sor              |  Successive over-relaxation

### Resources

- [Cahn-Hilliard Example in FEniCS](https://fenicsproject.org/docs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html)

- [*Automated Solution of Differential Equations by the Finite Element Method (The FEniCS Book)*](https://fenicsproject.org/book/) -- [PDF](http://launchpad.net/fenics-book/trunk/final/+download/fenics-book-2011-10-27-final.pdf) available from the authors

- [Unified Form Language (UFL)](https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html)

