"""Regression tests for optional 3D mesh utilities."""

from types import SimpleNamespace

import pyvista as pv

from spateo.tdr.models.models_individual.mesh_utils import fix_mesh


def test_fix_mesh_supports_current_pymeshfix_repair_signature(monkeypatch):
    """MeshFix 0.18 removed the former ``verbose`` keyword."""

    class CurrentMeshFix:
        def __init__(self, mesh):
            self.mesh = mesh

        def repair(self):
            self.mesh = self.mesh.triangulate().clean()

    monkeypatch.setitem(__import__("sys").modules, "pymeshfix", SimpleNamespace(MeshFix=CurrentMeshFix))
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()

    fixed = fix_mesh(mesh)

    assert fixed.n_points > 0
    assert fixed.n_cells > 0
