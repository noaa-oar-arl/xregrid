import time
import numpy as np
import xarray as xr
import xesmf as xe
from xregrid import Regridder


def create_sample_dataset(nlat, nlon, ntime=1):
    """Create a sample dataset with synthetic data."""
    lat = np.linspace(-89.9, 89.9, nlat)
    lon = np.linspace(0, 359.5, nlon)
    time = np.arange(ntime)

    # Use float32 to save memory
    data = np.random.rand(ntime, nlat, nlon).astype(np.float32)

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": (["time"], time), "lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    return ds


def benchmark_resolution(
    name, nlat_in, nlon_in, nlat_out, nlon_out, ntime=10, trials=3
):
    print(f"\n--- Benchmarking {name} ({ntime} time steps) ---")
    source_ds = create_sample_dataset(nlat_in, nlon_in, ntime=ntime)
    target_ds = create_sample_dataset(nlat_out, nlon_out, ntime=1)

    # --- Weight Generation ---
    print("Generating weights...")
    regridder_xesmf = xe.Regridder(
        source_ds.isel(time=0), target_ds.isel(time=0), method="bilinear", periodic=True
    )
    regridder_xregrid = Regridder(
        source_ds.isel(time=0), target_ds.isel(time=0), method="bilinear", periodic=True
    )

    # --- Weight Application ---
    print(f"Applying weights ({trials} trials)...")

    # xESMF
    times_xesmf = []
    # Warmup
    _ = regridder_xesmf(source_ds["temperature"].isel(time=slice(0, 1)))
    for _ in range(trials):
        start = time.perf_counter()
        # Single time step application
        for i in range(ntime):
            _ = regridder_xesmf(source_ds["temperature"].isel(time=i))
        times_xesmf.append(time.perf_counter() - start)
    avg_xesmf = np.mean(times_xesmf) / ntime

    # XRegrid
    times_xregrid = []
    # Warmup
    _ = regridder_xregrid(source_ds["temperature"].isel(time=slice(0, 1)))
    for _ in range(trials):
        start = time.perf_counter()
        # Single time step application
        for i in range(ntime):
            _ = regridder_xregrid(source_ds["temperature"].isel(time=i))
        times_xregrid.append(time.perf_counter() - start)
    avg_xregrid = np.mean(times_xregrid) / ntime

    print(
        f"App avg per time step - xESMF: {avg_xesmf:.6f}s, XRegrid: {avg_xregrid:.6f}s"
    )
    print(f"Application Speedup: {avg_xesmf / avg_xregrid:.1f}x")

    return {
        "name": name,
        "app_xesmf": avg_xesmf,
        "app_xregrid": avg_xregrid,
        "speedup": avg_xesmf / avg_xregrid,
    }


results = []
# 1.0° Global
results.append(benchmark_resolution("1.0° Global", 180, 360, 180, 360, ntime=20))

# 0.25° Global
results.append(benchmark_resolution("0.25° Global", 720, 1440, 720, 1440, ntime=3))

# 0.1° Global
results.append(
    benchmark_resolution("0.1° Global", 1800, 3600, 1800, 3600, ntime=1, trials=1)
)

print("\n\n" + "=" * 50)
print("FINAL RESULTS SUMMARY (SINGLE TIME STEP REGRIDDING)")
print("=" * 50)
print(
    f"{'Resolution':<15} | {'xESMF App (s)':<15} | {'XRegrid App (s)':<15} | {'Speedup':<10}"
)
print("-" * 65)
for r in results:
    print(
        f"{r['name']:<15} | {r['app_xesmf']:<15.6f} | {r['app_xregrid']:<15.6f} | {r['speedup']:<10.1f}x"
    )
