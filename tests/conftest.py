import sys
import numpy as np


def setup_esmpy_mock():
    # Use real classes for the mock to avoid pickling recursion issues
    # and ensure isinstance works.

    class MockGrid:
        def __init__(self, *args, **kwargs):
            self.staggerloc = [0, 1]
            # args[0] is max_index (numpy array)
            if args and isinstance(args[0], np.ndarray):
                self.max_index = args[0]
            else:
                self.max_index = np.array([36, 18])
            self.coords = {}
            self.items = {}

        def get_coords(self, coord_dim, staggerloc=0):
            key = (coord_dim, staggerloc)
            if key not in self.coords:
                shape = list(self.max_index)
                if staggerloc == 1:  # CORNER
                    shape = [s + 1 for s in shape]
                self.coords[key] = np.zeros(tuple(shape))
            return self.coords[key]

        def get_item(self, item, staggerloc=0):
            key = (item, staggerloc)
            if key not in self.items:
                shape = list(self.max_index)
                if staggerloc == 1:  # CORNER
                    shape = [s + 1 for s in shape]
                self.items[key] = np.zeros(tuple(shape))
            return self.items[key]

        def add_item(self, item, staggerloc=0):
            pass

        def destroy(self):
            pass

    class MockLocStream:
        def __init__(self, *args, **kwargs):
            self.items = {}
            # location_count is a keyword arg
            self.size = kwargs.get("location_count", 10)

        def __setitem__(self, key, value):
            self.items[key] = value

        def __getitem__(self, key):
            if key not in self.items:
                self.items[key] = np.zeros(self.size)
            return self.items[key]

        def destroy(self):
            pass

    class MockMesh:
        def __init__(self, *args, **kwargs):
            self.nodes = []
            self.elements = []

        def add_nodes(self, *args, **kwargs):
            pass

        def add_elements(self, *args, **kwargs):
            pass

        def destroy(self):
            pass

    class MockField:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "field")
            self.grid = args[0] if args else None

        def destroy(self):
            pass

    class MockRegrid:
        def __init__(self, *args, **kwargs):
            pass

        def get_factors(self):
            return (np.array([0]), np.array([0]))

        def get_weights_dict(self, deep_copy=True):
            return {
                "row_dst": np.array([1]),
                "col_src": np.array([1]),
                "weights": np.array([1.0]),
            }

        def destroy(self):
            pass

    class MockESMF:
        def __init__(self):
            self._is_mock = True

            class CoordSys:
                SPH_DEG = 1
                CART = 0

            self.CoordSys = CoordSys

            class StaggerLoc:
                CENTER = 0
                CORNER = 1

            self.StaggerLoc = StaggerLoc

            class GridItem:
                MASK = 1

            self.GridItem = GridItem

            class RegridMethod:
                BILINEAR = 0
                CONSERVE = 1
                NEAREST_STOD = 2
                NEAREST_DTOS = 3
                PATCH = 4

            self.RegridMethod = RegridMethod

            class UnmappedAction:
                IGNORE = 1

            self.UnmappedAction = UnmappedAction

            class ExtrapMethod:
                NEAREST_STOD = 0
                NEAREST_IDAVG = 1
                CREEP_FILL = 2

            self.ExtrapMethod = ExtrapMethod

            class MeshLoc:
                NODE = 0
                ELEMENT = 1

            self.MeshLoc = MeshLoc

            class MeshElemType:
                TRI = 1
                QUAD = 2

            self.MeshElemType = MeshElemType

            class NormType:
                FRACAREA = 0
                DSTAREA = 1

            self.NormType = NormType

            class LogKind:
                MULTI = 1

            self.LogKind = LogKind

            self.Grid = MockGrid
            self.LocStream = MockLocStream
            self.Mesh = MockMesh
            self.Field = MockField
            self.Regrid = MockRegrid
            self.__version__ = "8.6.0"

        def Manager(self, *args, **kwargs):
            class MockManager:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return MockManager()

        def pet_count(self):
            return 1

        def local_pet(self):
            return 0

    mock_esmpy = MockESMF()
    sys.modules["esmpy"] = mock_esmpy
    return mock_esmpy


try:
    import esmpy

    # Verify it's actually working
    if hasattr(esmpy, "_is_mock"):
        raise ImportError("Already mocked")
    esmpy.Manager(debug=False)
    HAS_REAL_ESMF = True
    print("\n--- Real ESMF detected in conftest.py ---")
except (ImportError, Exception) as e:
    HAS_REAL_ESMF = False
    print(f"\n--- Real ESMF NOT detected in conftest.py: {e} ---")
    setup_esmpy_mock()
