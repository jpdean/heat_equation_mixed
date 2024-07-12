import numpy as np
from dolfinx import mesh, fem
from ufl import inner, dx
from mpi4py import MPI
from scipy.interpolate import RBFInterpolator, interp1d


def norm_L2(comm, v):
    return np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM)
    )


def markers_to_meshtags(msh, tags, markers, dim):
    entities = [mesh.locate_entities_boundary(msh, dim, marker) for marker in markers]
    values = [np.full_like(entities, tag) for (tag, entities) in zip(tags, entities)]
    entities = np.hstack(entities, dtype=np.int32)
    values = np.hstack(values, dtype=np.intc)
    perm = np.argsort(entities)
    return mesh.meshtags(msh, dim, entities[perm], values[perm])
