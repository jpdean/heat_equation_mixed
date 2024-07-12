from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_vector_block,
)
import ufl
from ufl import div, grad, inner, dx, FacetNormal, dot, as_vector
import numpy as np
from petsc4py import PETSc
from utils import norm_L2, markers_to_meshtags


class BoundaryCondition:
    def __init__(self, type, value):
        self.type = type
        self.value = value


def manufacture_solution(u_e, t, kappa, rho, c_v):
    sigma_e = -kappa * grad(u_e)
    f = ufl.diff(rho * c_v * u_e, t) + div(sigma_e)
    return sigma_e, f


def solve(
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
):
    # Get MPI communicator
    comm = msh.comm

    # Function spaces for the dual and primal unknowns
    V = fem.functionspace(msh, ("Raviart-Thomas", k + 1))
    Q = fem.functionspace(msh, ("Discontinuous Lagrange", k))

    # Function space for boundary data
    W = fem.functionspace(msh, ("Lagrange", k + 1))

    # Define trial and test functions
    sigma, tau = ufl.TrialFunction(V), ufl.TestFunction(V)
    u, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    # u at previous time step
    u_n = fem.Function(Q)
    u_n.interpolate(u_0_expr)

    # Create integration measures
    ds = ufl.Measure("ds", subdomain_data=ft)

    # Define problem
    n = FacetNormal(msh)
    delta_t = fem.Constant(msh, t_e / num_time_steps)
    a = [
        [inner(1 / kappa * sigma, tau) * dx, -inner(u, div(tau)) * dx],
        [inner(div(sigma), q) * dx, inner(rho * c_v * u / delta_t, q) * dx],
    ]
    L = [
        inner(fem.Constant(msh, (0.0, 0.0)), tau) * dx,
        inner(f + rho * c_v * u_n / delta_t, q) * dx,
    ]

    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_cells = mesh.compute_incident_entities(
        msh.topology, boundary_facets, fdim, tdim
    )

    # Boundary conditions
    bcs = []
    bc_funcs = {}
    for tag, bc in boundary_conditions.items():
        if bc.type == "dirichlet":
            if isinstance(bc.value, fem.Constant):
                u_d = bc.value
            else:
                u_d = fem.Function(W)
                u_d.interpolate(bc.value, boundary_cells)
                bc_funcs[tag] = u_d
            L[0] -= inner(u_d, dot(tau, n)) * ds(tag)
        elif bc.type == "neumann":
            sigma_n = fem.Function(V)
            sigma_expr = fem.Expression(bc.value, V.element.interpolation_points())
            sigma_n.interpolate(sigma_expr, boundary_cells)
            bc_funcs[tag] = (sigma_n, sigma_expr)
            facets = ft.find(tag)
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(sigma_n, dofs))
        elif bc.type == "robin":
            u_r = fem.Function(W)
            u_r_expr = fem.Expression(bc.value[1], W.element.interpolation_points())
            u_r.interpolate(u_r_expr, boundary_cells)

            h_c = fem.Function(W)
            if isinstance(bc.value[0], ufl.core.expr.Expr):
                h_c_expr = fem.Expression(bc.value[0], W.element.interpolation_points())
            else:
                h_c_expr = bc.value[0]
            h_c.interpolate(h_c_expr, boundary_cells)

            bc_funcs[tag] = (u_r, u_r_expr)
            a[0][0] += inner(dot(sigma, n) / h_c, dot(tau, n)) * ds(tag)
            L[0] -= inner(u_r, dot(tau, n)) * ds(tag)
        else:
            raise RuntimeError("Boundary condition not supported")

    a = fem.form(a)
    L = fem.form(L)

    # Assemble matrix
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()

    # Create RHS vector
    b = create_vector_block(L)

    # Set up solver
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # Output files
    sigma_h = fem.Function(V)
    if write_to_file:
        # Since sigma_h is in a Raviart-Thomas space, we need to interpolate it into
        # a discontinuous Lagrange space to write it to file
        X = fem.functionspace(
            msh, ("Discontinuous Lagrange", k + 1, (msh.geometry.dim,))
        )
        sigma_vis = fem.Function(X)
        sigma_vis.interpolate(sigma_h)

        files = [
            io.VTXWriter(msh.comm, file_name, func, "BP4")
            for (file_name, func) in [("sigma_h.bp", sigma_vis), ("u_h.bp", u_n)]
        ]

        for file in files:
            file.write(t.expression().value)

    # Time stepping loop
    x = A.createVecRight()
    for n in range(num_time_steps):
        print(f"Time step {n + 1} of {num_time_steps}")
        t.expression().value += delta_t.value

        # Update BCs
        for tag, bc_func in bc_funcs.items():
            bc = boundary_conditions[tag]
            if bc.type == "dirichlet":
                bc_func.interpolate(bc.value, boundary_cells)
            else:
                bc_func[0].interpolate(bc_func[1], boundary_cells)

        # Assemble RHS
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector_block(b, L, a, bcs=bcs)

        # Solve
        ksp.solve(b, x)

        # Recover solution
        offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        sigma_h.x.array[:offset] = x.array_r[:offset]
        sigma_h.x.scatter_forward()
        u_n.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
        u_n.x.scatter_forward()

        # Write solution to file
        if write_to_file:
            sigma_vis.interpolate(sigma_h)
            for file in files:
                file.write(t.expression().value)

    if write_to_file:
        for file in files:
            file.close()

    return sigma_h, u_n


if __name__ == "__main__":
    # Number of elements in each direction
    n = 16
    # Polynomial degree
    k = 1
    # Initial time
    t_i = 0.0
    # End time
    t_e = 0.5
    # Number of time steps
    num_time_steps = 10
    # Choose whether to write to file
    write_to_file = True

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

    # Create mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n)

    # Define boundary tags
    boundaries = {"left": 1, "right": 2, "bottom": 3, "top": 4}

    # Create facet meshtags
    tdim = msh.topology.dim
    fdim = tdim - 1
    markers = [
        lambda x: np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[0], 1.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[1], 1.0),
    ]
    ft = markers_to_meshtags(msh, boundaries.values(), markers, fdim)

    # Material properties
    kappa = fem.Constant(msh, 2.5)
    rho = fem.Constant(msh, 1.4)
    c_v = fem.Constant(msh, 0.8)

    # Create source term
    x = ufl.SpatialCoordinate(msh)
    t = ufl.variable(fem.Constant(msh, t_i))  # Time
    u_e = u_e_expr(x, ufl)
    sigma_e, f = manufacture_solution(u_e, t, kappa, rho, c_v)

    # Create boundary condition data
    h_c = 2.0 + 0.1 * ufl.sin(ufl.pi * x[0])
    u_r = u_e - dot(sigma_e, as_vector((1, 0))) / h_c

    boundary_conditions = {
        boundaries["left"]: BoundaryCondition("dirichlet", u_d_expr),
        boundaries["right"]: BoundaryCondition("robin", (h_c, u_r)),
        boundaries["bottom"]: BoundaryCondition("dirichlet", u_d_expr),
        boundaries["top"]: BoundaryCondition("neumann", sigma_e),
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
    e_sigma = norm_L2(comm, sigma_e - sigma_h)
    e_u = norm_L2(comm, u_e - u_h)

    print(f"e_u = {e_u}")
    print(f"e_sigma = {e_sigma}")
