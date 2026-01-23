import sys
from unittest.mock import MagicMock

import numpy as np
import xarray as xr

# Mock esmpy
mock_esmpy = MagicMock()
mock_esmpy.CoordSys.SPH_DEG = 1
mock_esmpy.StaggerLoc.CENTER = 0
mock_esmpy.GridItem.MASK = 1
mock_esmpy.RegridMethod.BILINEAR = 0
mock_esmpy.UnmappedAction.IGNORE = 1
sys.modules["esmpy"] = mock_esmpy

# Mock matplotlib.pyplot
mock_plt = MagicMock()
mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])
sys.modules["matplotlib.pyplot"] = mock_plt
sys.modules["matplotlib"] = MagicMock()

# Add src to path to use xregrid
sys.path.insert(0, "src")
from xregrid import ESMPyRegridder  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Mock ESMF Regrid behavior
mock_regrid = MagicMock()


def mock_generate_weights(*args, **kwargs):
    # Determine n_src and n_dst based on grids
    # This is a bit complex to do genericly in mock, so we'll just return a 1-1 mapping for the sample sizes
    # sample sizes: src (45, 90), dst (60, 120)
    mock_regrid.get_weights_dict.return_value = {
        "row_dst": np.array([1]),
        "col_src": np.array([1]),
        "weights": np.array([1.0]),
    }
    return mock_regrid


mock_esmpy.Regrid.side_effect = mock_generate_weights

# --- User's Script ---


# Create a simple source dataset
def create_sample_dataset(
    name, nlat=45, nlon=90, lat_range=(-90, 90), lon_range=(0, 360)
):
    """Create a sample dataset with synthetic data."""
    lat = np.linspace(lat_range[0], lat_range[1], nlat)
    lon = np.linspace(lon_range[0], lon_range[1], nlon)

    # Create some interesting synthetic data
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    data = np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid * 2))

    ds = xr.Dataset(
        {
            "temperature": (["lat", "lon"], data),
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
        }
    )
    return ds


# Create source and target grids
print("Creating source and target grids...")
source_ds = create_sample_dataset("source", nlat=45, nlon=90)
target_ds = create_sample_dataset("target", nlat=60, nlon=120)

print(f"Source grid shape: {source_ds.temperature.shape}")
print(f"Target grid shape: {target_ds.temperature.shape}")

# Create the regridder
print("Creating regridder...")
regridder = ESMPyRegridder(source_ds, target_ds, method="bilinear")

# Perform the regridding
print("Performing regridding...")
regridded_temp = regridder(source_ds["temperature"])

print(f"Regridded data shape: {regridded_temp.shape}")
print(
    f"Original data range: [{source_ds.temperature.min().values:.3f}, {source_ds.temperature.max().values:.3f}]"
)
print(
    f"Regridded data range: [{regridded_temp.min().values:.3f}, {regridded_temp.max().values:.3f}]"
)

# Plot the results
print("Mocking plot...")
# plt.subplots calls etc will work because of MagicMock

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data
im1 = axes[0].imshow(source_ds.temperature, aspect="auto", origin="lower")
axes[0].set_title("Original Data")
axes[0].set_xlabel("Longitude Index")
axes[0].set_ylabel("Latitude Index")
plt.colorbar(im1, ax=axes[0])

# Regridded data
im2 = axes[1].imshow(regridded_temp, aspect="auto", origin="lower")
axes[1].set_title("Regridded Data")
axes[1].set_xlabel("Longitude Index")
axes[1].set_ylabel("Latitude Index")
plt.colorbar(im2, ax=axes[1])

# Difference (interpolated back to source grid for comparison)
im3 = axes[2].imshow(regridded_temp, aspect="auto", origin="lower")
axes[2].set_title("Higher Resolution Result")
axes[2].set_xlabel("Longitude Index")
axes[2].set_ylabel("Latitude Index")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
# plt.show() # Skip show in test

print("Bilinear regridding completed successfully!")
