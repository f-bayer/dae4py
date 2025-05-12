# this example requires the following packages (might be incomplete)
# - recent FEniCSx: https://fenicsproject.org/download/
# - numpy
# - mpi4py
# - petsc4py
# - gmsh

# choose DAE index
# DAE_INDEX = 0
DAE_INDEX = 1

# problem = "lid cavity"
# problem = "channel"
problem = "von Karmann"

# --- Imports ---
import time
import typing
import numpy as np
import basix
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio, VTKFile
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
from dolfinx.io import VTKFile

from dolfinx import default_scalar_type

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp

from dae4py.radau import solve_dae_radau


class Domain:
    def __init__(
        self,
        mesh: dmesh.Mesh,
        facet_fkts: typing.Any = None,
        ds: typing.Any = None,
        quadrature_degree: typing.Optional[int] = None,
        dv: typing.Optional[typing.Any] = None,
        cell_tags=None,
        facet_tags=None,
    ):
        # Mesh
        self.mesh = mesh

        # Facet functions
        self.facet_functions = facet_fkts

        self.cell_tags = cell_tags
        self.facet_tags = facet_tags

        # Integrators
        if ds is None:
            self.ds = ufl.ds
        else:
            self.ds = ds

        if dv is None:
            if quadrature_degree is None:
                self.dv = ufl.dx
            else:
                self.dv = ufl.dx(degree=quadrature_degree)
        else:
            self.dv = dv


def create_rectangle_builtin(
    nelmt: typing.List[int], extends: typing.List[float] = [1.0, 1.0]
):
    # --- Create mesh
    ctype = dmesh.CellType.triangle
    dtype = dmesh.DiagonalType.left

    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array(extends)],
        [nelmt[0], nelmt[1]],
        cell_type=ctype,
        diagonal=dtype,
    )

    # --- Tag facets on final mesh
    # Define boundaries
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], extends[0])),
        (4, lambda x: np.isclose(x[1], extends[1])),
    ]

    # Set markers
    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dmesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = dmesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_function)

    return Domain(domain, facet_function, ds)


# references
# https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
# https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_gmsh.py
# https://jsdokken.com/src/tutorial_gmsh.html
# https://jsdokken.com/src/pygmsh_tutorial.html
# https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py
# https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html
def create_rectangle_with_hole_builtin(
    characteristic_length: float = 5e-2,
    extends: typing.List[float] = [2.2, 0.41],
    center: typing.List[float] = [0.2, 0.2],
    radius: float = 0.05,
    dim: int = 2,
):
    try:
        import gmsh
    except:
        raise ImportError(
            "'import gmsh' failed. You have to install gmsh to use this function."
        )

    # initialize gmsh and cerate model
    gmsh.initialize()
    model = gmsh.model()

    name = "rectangle_with_hole"
    model.add(name)
    model.setCurrent(name)

    # channel
    if dim == 2:
        channel = model.occ.addRectangle(0, 0, 0, *extends)
    elif dim == 3:
        channel = model.occ.addBox(0, 0, 0, *extends, 1)
    else:
        raise NotImplementedError

    # disk
    if dim == 2:
        disk = model.occ.addDisk(*center, 0, radius, radius)
    elif dim == 3:
        disk = model.occ.addSphere(*center, 0.5, radius)
    else:
        raise NotImplementedError

    # cut disk from the channel
    fluid = model.occ.cut([(dim, channel)], [(dim, disk)])

    model.occ.synchronize()

    # add volume
    volumes = model.getEntities(dim=dim)
    assert volumes == fluid[0]
    fluid_marker = 11
    model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")

    # add markers
    left_marker = 1
    bottom_marker = 2
    right_marker = 3
    top_marker = 4
    disk_marker = 5

    # Define boundaries
    boundaries = [
        (left_marker, lambda x: np.isclose(x[0], 0)),  # left
        (bottom_marker, lambda x: np.isclose(x[1], 0)),  # bottom
        (right_marker, lambda x: np.isclose(x[0], extends[0])),  # right
        (top_marker, lambda x: np.isclose(x[1], extends[1])),  # top
        (
            disk_marker,
            lambda x: np.logical_and(
                np.isclose(x[0], center[0]),
                np.isclose(x[1], center[1]),
            ),
        ),
    ]

    # mark boundaries
    left, bottom, right, top, disk = None, None, None, None, None
    for line in model.getEntities(dim=dim - 1):
        com = model.occ.getCenterOfMass(line[0], line[1])
        print(f"com: {com}")

        if boundaries[0][1](com):
            left = line[1]
        elif boundaries[1][1](com):
            bottom = line[1]
        elif boundaries[2][1](com):
            right = line[1]
        elif boundaries[3][1](com):
            top = line[1]
        elif boundaries[4][1](com):
            disk = line[1]
        else:
            raise RuntimeError("You should never be here")

    model.addPhysicalGroup(dim - 1, [left], left_marker, "left")
    model.addPhysicalGroup(dim - 1, [bottom], bottom_marker, "bottom")
    model.addPhysicalGroup(dim - 1, [right], right_marker, "right")
    model.addPhysicalGroup(dim - 1, [top], top_marker, "top")
    model.addPhysicalGroup(dim - 1, [disk], disk_marker, "disk")

    # Set the characteristic length for a finer mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length)

    # create mesh
    model.mesh.generate(dim=dim)

    # create dolfinx mesh
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        model, MPI.COMM_WORLD, rank=0, gdim=dim
    )
    mesh.name = name

    # create domain
    return Domain(mesh, cell_tags=cell_tags, facet_tags=facet_tags)


def dolfinx_to_scipy(A):
    A_csr = A.getValuesCSR()
    return csr_matrix(A_csr[::-1], shape=A.getSize())


if __name__ == "__main__":
    match problem:
        case "lid cavity":
            N = 20
            extends = [1.2, 1.0]
            domain = create_rectangle_builtin([N, N], extends)
        case "channel":
            N = 10
            extends = [2.2, 0.41]
            domain = create_rectangle_builtin([5 * N, N], extends)
        case "von Karmann":
            # see https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_gmsh.py
            # how to use gmsh to create a mesh for the problem of rectangle - circle
            characteristic_length = 5e-2  # original resolution
            # characteristic_length = 2.5e-2 # ~600s
            extends = [2.2, 0.41]
            domain = create_rectangle_with_hole_builtin(characteristic_length, extends)
        case _:
            raise NotImplementedError

    gdim = domain.mesh.geometry.dim
    n = ufl.FacetNormal(domain.mesh)  # normals

    # function spaces
    k = 1
    Pu = basix.ufl.element("Lagrange", domain.mesh.basix_cell(), k + 1, shape=(gdim,))
    Pp = basix.ufl.element("Lagrange", domain.mesh.basix_cell(), k)

    V = dfem.functionspace(domain.mesh, basix.ufl.mixed_element([Pu, Pp]))

    # trial and test functions
    u, p = ufl.TrialFunctions(V)
    v_u, v_p = ufl.TestFunctions(V)

    w = dfem.Function(V)
    u_h, p_h = ufl.split(w)
    wp = dfem.Function(V)
    up_h, pp_h = ufl.split(wp)

    # nonlinear form
    rho = 1
    match problem:
        case "lid cavity":
            Re = 300
        case "channel":
            Re = 1000
        case "von Karmann":
            Re = 1000

    mu = 1 / Re

    match DAE_INDEX:
        case 0:
            # rhs form
            F = (
                # momentum equation
                +mu * ufl.inner(ufl.grad(v_u), ufl.grad(u_h)) * ufl.dx
                + rho * ufl.inner(v_u, ufl.grad(u_h) * u_h) * ufl.dx
            )

            # mass matrix form
            am = (
                # du/dt
                ufl.inner(v_u, u) * ufl.dx
                # pressure term
                - ufl.inner(ufl.div(v_u), p) * ufl.dx
                # incompressibility
                + ufl.inner(v_p, ufl.div(u)) * ufl.dx
            )
        case 1:
            # rhs form
            F = (
                # momentum equation
                +mu * ufl.inner(ufl.grad(v_u), ufl.grad(u_h)) * ufl.dx
                + rho * ufl.inner(v_u, ufl.grad(u_h) * u_h) * ufl.dx
                # incompressibility
                + ufl.inner(v_p, ufl.div(u_h)) * ufl.dx
            )

            # mass matrix form
            am = (
                # du/dt
                ufl.inner(v_u, u) * ufl.dx
                # pressure term
                - ufl.inner(ufl.div(v_u), p) * ufl.dx
            )
        case _:
            raise NotImplementedError

    # Jacobian of rhs
    J = ufl.derivative(F, w)

    # dirichlet boundary conditions
    fct_fkts = domain.facet_functions

    # initialize list
    bcs = []

    # facet-cell connectivity
    domain.mesh.topology.create_connectivity(1, 2)

    ################################################
    # dirichlet boundary conditions for the velocity
    ################################################
    match problem:
        case "lid cavity":
            # - walls (no slip)
            V_u, _ = V.sub(0).collapse()
            wall_velocity = dfem.Function(V_u)
            wall_velocity.x.array[:] = 0.0
            fcts = fct_fkts.indices[
                np.logical_or.reduce(
                    (
                        (fct_fkts.values == 1),  # left
                        (fct_fkts.values == 2),  # bottom
                        (fct_fkts.values == 3),  # right
                        # (fct_fkts.values == 4), # top
                    )
                )
            ]
            dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
            bcs.append(dfem.dirichletbc(wall_velocity, dofs, V.sub(0)))

            # - driving (lid) velocity condition on top boundary
            lid_velocity = dfem.Function(V_u)

            def smoothstep2(x, x_min=0, x_max=1):
                x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
                return 6 * x**5 - 15 * x**4 + 10 * x**3

            u0 = lambda t: smoothstep2(t, 0, 5)

            def inflow_profile(x, t=0):
                return np.stack(
                    (
                        u0(t) * np.ones(x.shape[1]),
                        np.zeros(x.shape[1]),
                    )
                )

            lid_velocity.interpolate(inflow_profile)
            fcts = fct_fkts.indices[
                np.logical_or.reduce(
                    (
                        # (fct_fkts.values == 1), # left
                        # (fct_fkts.values == 2), # bottom
                        # (fct_fkts.values == 3), # right
                        (fct_fkts.values == 4),  # top
                    )
                )
            ]
            dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
            bcs.append(dfem.dirichletbc(lid_velocity, dofs, V.sub(0)))

        case "channel":
            # - walls (no slip)
            V_u, _ = V.sub(0).collapse()
            wall_velocity = dfem.Function(V_u)
            wall_velocity.x.array[:] = 0.0
            fcts = fct_fkts.indices[
                np.logical_or.reduce(
                    (
                        # (fct_fkts.values == 1), # left
                        (fct_fkts.values == 2),  # bottom
                        # (fct_fkts.values == 3), # right
                        (fct_fkts.values == 4),  # top
                    )
                )
            ]
            dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
            bcs.append(dfem.dirichletbc(wall_velocity, dofs, V.sub(0)))

            # - inflow
            def inflow_profile(x):
                return np.stack(
                    (
                        4.0 * x[1] * (extends[1] - x[1]) / extends[1] ** 2,
                        np.zeros(x.shape[1]),
                    )
                )

            lid_velocity = dfem.Function(V_u)
            lid_velocity.interpolate(inflow_profile)
            fcts = fct_fkts.indices[
                np.logical_or.reduce(
                    (
                        (fct_fkts.values == 1),  # left
                        # (fct_fkts.values == 2), # bottom
                        # (fct_fkts.values == 3), # right
                        # (fct_fkts.values == 4), # top
                    )
                )
            ]
            dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
            bcs.append(dfem.dirichletbc(lid_velocity, dofs, V.sub(0)))

        case "von Karmann":
            # - walls (no slip)
            V_u, _ = V.sub(0).collapse()
            wall_velocity = dfem.Function(V_u)
            wall_velocity.x.array[:] = 0.0
            walls_tag = [2, 4, 5]
            for tag in walls_tag:
                fcts = domain.facet_tags.find(tag)
                dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
                bcs.append(dfem.dirichletbc(wall_velocity, dofs, V.sub(0)))

            # - inflow
            # https://ferrite-fem.github.io/Ferrite.jl/stable/examples/ns_vs_diffeq/
            lid_velocity = dfem.Function(V_u)

            def smoothstep2(x, x_min=0, x_max=1):
                x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
                return 6 * x**5 - 15 * x**4 + 10 * x**3

            u0 = lambda t: 1.5 * smoothstep2(t)

            def inflow_profile(x, t=0):
                return np.stack(
                    (
                        4.0 * u0(t) * x[1] * (extends[1] - x[1]) / extends[1] ** 2,
                        np.zeros(x.shape[1]),
                    )
                )

            lid_velocity.interpolate(inflow_profile)
            fcts = domain.facet_tags.find(1)
            dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, fcts)
            bcs.append(dfem.dirichletbc(lid_velocity, dofs, V.sub(0)))
        case _:
            raise NotImplementedError

    # assembly and conversion to numpy
    form_am = dfem.form(am)
    residual = dfem.form(F)
    jacobian = dfem.form(J)

    # create the sparse matrix and vector containing the residual only once.
    M = dfem_petsc.create_matrix(form_am)
    A = dfem_petsc.create_matrix(jacobian)
    L = dfem_petsc.create_vector(residual)

    # assemble mass matrix only once, include boundary conditions and compute
    # its factorization after conversation to scipy sparse matrix
    dfem_petsc.assemble_matrix(M, form_am, bcs=bcs)
    M.assemble()
    M_scipy = dolfinx_to_scipy(M)
    if DAE_INDEX == 0:
        M_factor = splu(M_scipy)

    def fun_ode(t, y):
        if problem == "lid cavity":
            lid_velocity.interpolate(lambda x: inflow_profile(x, t))
        if problem == "von Karmann":
            lid_velocity.interpolate(lambda x: inflow_profile(x, t))

        # assign function value
        w.vector.setArray(y)

        # assemble Jacobian and residual
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()
        dfem_petsc.assemble_matrix(A, jacobian, bcs=bcs)
        A.assemble()
        dfem_petsc.assemble_vector(L, residual)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        # compute b - J(u_D-u_(i-1))
        dfem_petsc.apply_lifting(L, [jacobian], [bcs], x0=[w.vector], scale=1)
        # set du|_bc = u_{i-1}-u_D
        dfem_petsc.set_bc(L, bcs, w.vector, 1.0)
        L.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        return M_factor.solve(L.array)

    def fun_dae(t, y, yp):
        # print(f"t: {t}")
        if problem == "lid cavity":
            lid_velocity.interpolate(lambda x: inflow_profile(x, t))
        if problem == "von Karmann":
            lid_velocity.interpolate(lambda x: inflow_profile(x, t))

        # assign function value
        w.x.array[:] = y
        wp.x.array[:] = yp

        # assemble Jacobian and residual
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()
        dfem_petsc.assemble_matrix(A, jacobian, bcs=bcs)
        A.assemble()
        dfem_petsc.assemble_vector(L, residual)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        # compute b - J(u_D-u_(i-1))
        dfem_petsc.apply_lifting(L, [jacobian], [bcs], x0=[w.x.petsc_vec])
        # set du|_bc = u_{i-1}-u_D
        dfem_petsc.set_bc(L, bcs, w.x.petsc_vec, 1.0)
        L.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        return M_scipy @ yp - L.array

    def jac_dae(t, y, yp):
        # assign function value
        # w.x.petsc_vec.setArray(y)
        # wp.x.petsc_vec.setArray(yp)
        w.x.array[:] = y
        wp.x.array[:] = yp

        # assemble Jacobian and residual
        A.zeroEntries()
        dfem_petsc.assemble_matrix(A, jacobian, bcs=bcs)
        A.assemble()

        return M_scipy, dolfinx_to_scipy(A)

    # initial conditions
    t0 = 0
    w.x.array[:] = 0.0
    wp.x.array[:] = 0.0
    y0 = w.x.array[:]
    yp0 = wp.x.array[:]

    # solver setup
    h0 = 1e-5
    atol = 1e-3
    rtol = 1e-3
    t1 = 10
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=500)

    # DAE solver
    jac = jac_dae
    start = time.time()
    sol = solve_dae_radau(
        fun_dae,
        y0,
        yp0,
        t_span,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        h0=h0,
        jac=jac,
        s=3,
        newton_iter_embedded=0,
    )
    end = time.time()
    print(f"elapsed time: {end - start}")

    # vtk export
    with (
        VTKFile(domain.mesh.comm, "fenics/u.pvd", "w") as file_u,
        VTKFile(domain.mesh.comm, "fenics/p.pvd", "w") as file_p,
    ):
        for ti, yi in zip(sol.t, sol.y):
            # w.vector.setArray(yi)
            w.x.petsc_vec.setArray(yi)
            u, p = w.sub(0).collapse(), w.sub(1).collapse()
            u.name = "u"
            p.name = "p"

            file_u.write_function(u, ti)
            file_p.write_function(p, ti)
