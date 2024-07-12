from mpi4py import MPI
from solver import solve, manufacture_solution, BoundaryCondition
import ufl
import numpy as np
from dolfinx import mesh, fem, io
import pytest
from utils import norm_L2, markers_to_meshtags
import gmsh


def compute_convergence_rate(ns, es):
    return np.log(es[0] / es[1]) / np.log(ns[1] / ns[0])


def create_mesh_fenics(comm, n, boundaries):
    # Create mesh
    msh = mesh.create_unit_square(comm, n, n)

    # Create facet meshtags
    tdim = msh.topology.dim
    fdim = tdim - 1
    markers = [
        lambda x: np.isclose(x[1], 1.0) | np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[0], 1.0),
    ]
    ft = markers_to_meshtags(msh, boundaries.values(), markers, fdim)

    return msh, ft


def create_mesh_gmsh(comm, n, boundaries):
    h = 1 / n
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("model")
        factory = gmsh.model.geo

        points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 0.0, 0.0, h),
            factory.addPoint(1.0, 1.0, 0.0, h),
            factory.addPoint(0.0, 1.0, 0.0, h),
        ]

        lines = [
            factory.addLine(points[0], points[1]),
            factory.addLine(points[1], points[2]),
            factory.addLine(points[2], points[3]),
            factory.addLine(points[3], points[0]),
        ]

        curves = [factory.addCurveLoop(lines)]

        surfaces = [factory.addPlaneSurface(curves)]

        factory.synchronize()

        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.addPhysicalGroup(1, [lines[2], lines[3]], boundaries["top_left"])
        gmsh.model.addPhysicalGroup(1, [lines[0]], boundaries["bottom"])
        gmsh.model.addPhysicalGroup(1, [lines[1]], boundaries["right"])

        gmsh.model.mesh.generate(2)
        # gmsh.fltk.run()
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2, partitioner=partitioner
    )
    gmsh.finalize()
    return msh, ft


@pytest.mark.parametrize("create_mesh", [create_mesh_fenics, create_mesh_gmsh])
@pytest.mark.parametrize("k", [i for i in range(3)])
def test_convergence(create_mesh, k):
    def u_e_expr(x, m=np):
        return m.sin(m.pi * x[0]) * m.cos(m.pi * x[1])

    def u_d_expr(x):
        return u_e_expr(x, np)

    def u_0_expr(x):
        return u_e_expr(x, np)

    # Number of elements in each direction
    ns = [16 * i for i in range(1, 3)]
    comm = MPI.COMM_WORLD
    # End time
    t_e = 1e12
    # Number of time steps
    num_time_steps = 1
    # Choose whether to write to file
    write_to_file = False

    # Define boundary markers
    boundaries = {"top_left": 1, "bottom": 2, "right": 3}

    e_sigma = []
    e_u = []
    for n in ns:
        msh, ft = create_mesh(comm, n, boundaries)

        # Material properties
        x = ufl.SpatialCoordinate(msh)
        kappa = 2.5 + 0.1 * ufl.sin(ufl.pi * x[0])
        rho = fem.Constant(msh, 1.4)
        c_v = fem.Constant(msh, 0.8)

        # Create source term
        t = ufl.variable(fem.Constant(msh, 0.0))  # Time
        u_e = u_e_expr(x, ufl)
        sigma_e, f = manufacture_solution(u_e, t, kappa, rho, c_v)

        # Create boundary condition data
        h_c = 2.0 + 0.1 * ufl.sin(ufl.pi * x[0])
        u_r = u_e - ufl.dot(sigma_e, ufl.as_vector((1, 0))) / h_c

        boundary_conditions = {
            boundaries["top_left"]: BoundaryCondition("dirichlet", u_d_expr),
            boundaries["bottom"]: BoundaryCondition("neumann", sigma_e),
            boundaries["right"]: BoundaryCondition("robin", (h_c, u_r)),
        }

        # Solve problem
        sigma_h, u_h = solve(
            msh,
            k,
            f,
            ft,
            boundary_conditions,
            kappa,
            rho,
            c_v,
            t_e,
            num_time_steps,
            write_to_file,
            t,
            u_0_expr,
        )

        # Compute the L2-norm of the error in the solution
        e_sigma.append(norm_L2(comm, sigma_e - sigma_h))
        e_u.append(norm_L2(comm, u_e_expr(x, ufl) - u_h))

    r_sigma = compute_convergence_rate(ns, e_sigma)
    r_u = compute_convergence_rate(ns, e_u)

    assert np.isclose(k + 1, r_sigma, atol=0.1)
    assert np.isclose(k + 1, r_u, atol=0.1)


def test_convergence_time():
    # Number of elements in each direction
    n = 16
    # Polynomial degree
    k = 3
    # Initial time
    t_i = 0.1
    # End time
    t_e = 0.5
    # Number of time steps
    num_time_steps = [16 * i for i in range(1, 3)]
    # Choose whether to write to file
    write_to_file = False

    def u_e_expr(x, m=np):
        if m == np:
            _t = t.expression().value
        else:
            _t = t
        return m.sin(m.pi * x[0]) * m.cos(m.pi * x[1]) * m.sin(m.pi * _t)

    def u_d_expr(x):
        return u_e_expr(x, np)

    def u_0_expr(x):
        return u_e_expr(x, np)

    # Define boundary markers
    boundaries = {"top_left": 1, "bottom": 2, "right": 3}

    # Create mesh
    comm = MPI.COMM_WORLD
    msh, ft = create_mesh_fenics(comm, n, boundaries)

    # Material properties
    x = ufl.SpatialCoordinate(msh)
    kappa = 2.5 + 0.1 * ufl.sin(ufl.pi * x[0])
    rho = fem.Constant(msh, 1.4)
    c_v = fem.Constant(msh, 0.8)

    # Create source term
    t = ufl.variable(fem.Constant(msh, t_i))  # Time
    u_e = u_e_expr(x, ufl)
    sigma_e, f = manufacture_solution(u_e, t, kappa, rho, c_v)

    # Create boundary condition data
    h_c = fem.Constant(msh, 2.0)
    u_r = u_e - ufl.dot(sigma_e, ufl.as_vector((1, 0))) / h_c

    boundary_conditions = {
        boundaries["top_left"]: BoundaryCondition("dirichlet", u_d_expr),
        boundaries["bottom"]: BoundaryCondition("neumann", sigma_e),
        boundaries["right"]: BoundaryCondition("robin", (h_c, u_r)),
    }

    e_sigma = []
    e_u = []
    for steps in num_time_steps:
        t.expression().value = t_i

        # Solve problem
        sigma_h, u_h = solve(
            msh,
            k,
            f,
            ft,
            boundary_conditions,
            kappa,
            rho,
            c_v,
            t_e,
            steps,
            write_to_file,
            t,
            u_0_expr,
        )

        # Compute the L2-norm of the error in the solution
        e_sigma.append(norm_L2(comm, sigma_e - sigma_h))
        e_u.append(norm_L2(comm, u_e_expr(x, ufl) - u_h))

    r_sigma = compute_convergence_rate(num_time_steps, e_sigma)
    r_u = compute_convergence_rate(num_time_steps, e_u)

    assert np.isclose(1, r_sigma, atol=0.1)
    assert np.isclose(1, r_u, atol=0.1)
