"""Shared fixtures."""

import pytest

from voxelmetry.phantom import sphere_phantom, torso_phantom


@pytest.fixture(scope="session")
def sphere():
    """A 30 mm-radius sphere on a 1 mm isotropic grid, with its label map."""
    return sphere_phantom(radius_mm=30.0, shape=(80, 80, 80), spacing=(1.0, 1.0, 1.0))


@pytest.fixture(scope="session")
def torso():
    """An anisotropic torso phantom: organ, lesions and bone."""
    return torso_phantom(shape=(96, 128, 128), spacing=(2.5, 1.5, 1.5), seed=0)
