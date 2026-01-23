"""
Weight Reuse and Performance Optimization
=========================================

This example demonstrates the most important performance optimization in XRegrid:
weight reuse. By saving and reloading regridding weights, you can achieve
massive speedups when regridding multiple datasets with the same grids.

This is particularly important for:
- Processing time series of data
- Operational forecast systems
- Climate data analysis workflows
- Any repeated regridding operations

Key concepts demonstrated:
- Saving and loading regridding weights
- Performance comparison with/without weight reuse
- Optimal workflow patterns
- Memory and disk usage considerations
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for gallery
import matplotlib.pyplot as plt
import time  # noqa: E402
import os
from xregrid import ESMPyRegridder

print("Weight Reuse and Performance Optimization")
print("=" * 50)

# Create representative grids for a common use case:
# Regridding from 0.25° climate model output to 1° analysis grid

# Source: 0.25° global grid (typical high-res climate model)
source_lats = np.linspace(-90, 90, 721)  # 0.25° spacing
source_lons = np.linspace(0, 359.75, 1440)  # 0.25° spacing

source_grid = xr.Dataset({"lat": (["lat"], source_lats), "lon": (["lon"], source_lons)})

# Target: 1° global grid (typical analysis resolution)
target_lats = np.linspace(-90, 90, 181)  # 1° spacing
target_lons = np.linspace(0, 359, 360)  # 1° spacing

target_grid = xr.Dataset({"lat": (["lat"], target_lats), "lon": (["lon"], target_lons)})

print(f"Source grid: {len(source_lats)}x{len(source_lons)} (0.25° resolution)")
print(f"Target grid: {len(target_lats)}x{len(target_lons)} (1° resolution)")
print(
    f"Grid size ratio: {len(source_lats)*len(source_lons) / (len(target_lats)*len(target_lons)):.1f}x"
)


# Create synthetic climate data with realistic patterns
def create_climate_data(lats, lons, n_time_steps=24):
    """Create realistic climate data (temperature) with temporal evolution"""

    # Create 2D coordinate arrays
    lons_2d, lats_2d = np.meshgrid(lons, lats)

    # Base temperature pattern
    # - Strong latitudinal gradient
    # - Land-ocean contrasts
    # - Topographic effects
    base_temp = 15 + 25 * np.cos(np.radians(lats_2d))

    # Add continental effects (simplified)
    continental = np.zeros_like(lons_2d)
    # Eurasia
    continental += -8 * np.exp(
        -((lons_2d - 90) ** 2 / 3000 + (lats_2d - 60) ** 2 / 500)
    )
    # North America
    continental += -6 * np.exp(
        -((lons_2d - 260) ** 2 / 2000 + (lats_2d - 50) ** 2 / 400)
    )

    # Create time series (2 years of monthly data)
    time_data = np.zeros((n_time_steps, len(lats), len(lons)))

    for t in range(n_time_steps):
        # Seasonal cycle (stronger in NH)
        month = t % 12
        seasonal = (
            15 * np.cos(2 * np.pi * (month - 6) / 12) * np.maximum(0, lats_2d / 90)
        )

        # Interannual variability
        year = t // 12
        interannual = 2 * np.sin(2 * np.pi * year / 5) * np.exp(-(lats_2d**2 / 2000))

        # Weather noise
        np.random.seed(42 + t)  # Different but reproducible for each time
        weather_noise = np.random.normal(0, 3, lats_2d.shape)

        time_data[t] = base_temp + continental + seasonal + interannual + weather_noise

    return time_data


# Create sample dataset
n_time_steps = 24  # 2 years of monthly data
print(f"\nCreating synthetic climate dataset ({n_time_steps} time steps)...")

temperature_data = create_climate_data(source_lats, source_lons, n_time_steps)

# Create xarray Dataset
dataset = xr.Dataset(
    {
        "temperature": (["time", "lat", "lon"], temperature_data),
        "lat": (["lat"], source_lats),
        "lon": (["lon"], source_lons),
        "time": pd.date_range("2020-01-01", periods=n_time_steps, freq="M"),
    }
)

print(f"Created dataset with shape: {dataset.temperature.shape}")
print(f"Data size: {dataset.temperature.nbytes / 1024**2:.1f} MB")

# Performance test 1: First-time weight generation
print("\n" + "=" * 50)
print("PERFORMANCE TEST 1: Weight Generation")
print("=" * 50)

weight_file = "performance_test_weights.nc"

# Clean up any existing weight file
if os.path.exists(weight_file):
    os.remove(weight_file)

start_time = time.time()
regridder_generate = ESMPyRegridder(
    source_grid,
    target_grid,
    method="bilinear",
    periodic=True,
    reuse_weights=True,
    filename=weight_file,
)
generation_time = time.time() - start_time

print(f"Weight generation time: {generation_time:.2f} seconds")
print(f"Weight file size: {os.path.getsize(weight_file) / 1024**2:.1f} MB")

# Performance test 2: Weight loading
print("\n" + "=" * 50)
print("PERFORMANCE TEST 2: Weight Loading")
print("=" * 50)

start_time = time.time()
regridder_load = ESMPyRegridder(
    source_grid,
    target_grid,
    method="bilinear",
    periodic=True,
    reuse_weights=True,
    filename=weight_file,
)
loading_time = time.time() - start_time

print(f"Weight loading time: {loading_time:.3f} seconds")
print(f"Speedup from weight reuse: {generation_time / loading_time:.1f}x")

# Performance test 3: Regridding application
print("\n" + "=" * 50)
print("PERFORMANCE TEST 3: Regridding Application")
print("=" * 50)

# Test with single time step
single_time_data = dataset.temperature.isel(time=0)

# Time multiple applications
n_iterations = 10
start_time = time.time()
for _ in range(n_iterations):
    result = regridder_load(single_time_data)
end_time = time.time()

single_regrid_time = (end_time - start_time) / n_iterations

print(f"Single time step regridding: {single_regrid_time:.4f} seconds")
print(f"Points processed per second: {single_time_data.size / single_regrid_time:,.0f}")

# Test with full time series
start_time = time.time()
full_result = regridder_load(dataset.temperature)
full_regrid_time = time.time() - start_time

print(f"Full time series regridding: {full_regrid_time:.2f} seconds")
print(f"Time per time step: {full_regrid_time / n_time_steps:.4f} seconds")
print(
    f"Vectorization speedup: {single_regrid_time / (full_regrid_time / n_time_steps):.1f}x"
)

# Memory usage analysis
print("\n" + "=" * 50)
print("MEMORY USAGE ANALYSIS")
print("=" * 50)

input_size_mb = dataset.temperature.nbytes / 1024**2
output_size_mb = full_result.nbytes / 1024**2

print(f"Input data size: {input_size_mb:.1f} MB")
print(f"Output data size: {output_size_mb:.1f} MB")
print(f"Memory expansion: {output_size_mb / input_size_mb:.1f}x")

# Workflow comparison
print("\n" + "=" * 50)
print("WORKFLOW COMPARISON")
print("=" * 50)


def workflow_without_reuse(data):
    """Simulate processing without weight reuse (BAD practice)"""
    total_time = 0
    results = []

    for t in range(len(data.time)):
        start = time.time()
        # Create new regridder each time (inefficient!)
        temp_regridder = ESMPyRegridder(
            source_grid, target_grid, method="bilinear", periodic=True
        )
        result = temp_regridder(data.isel(time=t))
        total_time += time.time() - start
        results.append(result)

    return results, total_time


def workflow_with_reuse(data, regridder):
    """Simulate processing with weight reuse (GOOD practice)"""
    start = time.time()
    result = regridder(data)
    total_time = time.time() - start
    return result, total_time


# Test both workflows (use subset for speed)
test_data = dataset.temperature.isel(time=slice(0, 6))  # 6 months

print("Testing workflow without weight reuse (6 time steps)...")
results_bad, time_bad = workflow_without_reuse(test_data)

print("Testing workflow with weight reuse (6 time steps)...")
result_good, time_good = workflow_with_reuse(test_data, regridder_load)

print("\nWorkflow comparison:")
print(f"Without reuse: {time_bad:.2f} seconds")
print(f"With reuse:    {time_good:.2f} seconds")
print(f"Speedup:       {time_bad / time_good:.1f}x")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Weight Reuse and Performance Optimization", fontsize=16)

# Performance comparison bar chart
methods = [
    "Weight\nGeneration",
    "Weight\nLoading",
    "Single\nRegridding",
    "Full Series\n(per step)",
]
times = [
    generation_time,
    loading_time,
    single_regrid_time,
    full_regrid_time / n_time_steps,
]

bars = axes[0, 0].bar(methods, times, color=["red", "green", "blue", "orange"])
axes[0, 0].set_ylabel("Time (seconds)")
axes[0, 0].set_title("Performance Breakdown")
axes[0, 0].set_yscale("log")

# Add value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{time_val:.3f}s",
        ha="center",
        va="bottom",
    )

# Memory usage pie chart
memory_labels = ["Input Data", "Output Data"]
memory_sizes = [input_size_mb, output_size_mb - input_size_mb]
colors = ["lightblue", "lightcoral"]

wedges, texts, autotexts = axes[0, 1].pie(
    memory_sizes, labels=memory_labels, autopct="%1.1f MB", colors=colors, startangle=90
)
axes[0, 1].set_title("Memory Usage")

# Workflow comparison
workflow_methods = ["Without\nReuse", "With\nReuse"]
workflow_times = [time_bad, time_good]

bars2 = axes[0, 2].bar(workflow_methods, workflow_times, color=["red", "green"])
axes[0, 2].set_ylabel("Time (seconds)")
axes[0, 2].set_title("Workflow Comparison (6 time steps)")

for bar, time_val in zip(bars2, workflow_times):
    height = bar.get_height()
    axes[0, 2].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{time_val:.2f}s",
        ha="center",
        va="bottom",
    )

# Show original and regridded data
im1 = axes[1, 0].pcolormesh(
    source_lons,
    source_lats,
    dataset.temperature.isel(time=0),
    shading="auto",
    cmap="RdYlBu_r",
)
axes[1, 0].set_title("Original 0.25° Data (January)")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[1, 0], label="Temperature (°C)")

im2 = axes[1, 1].pcolormesh(
    target_lons, target_lats, full_result.isel(time=0), shading="auto", cmap="RdYlBu_r"
)
axes[1, 1].set_title("Regridded 1° Data (January)")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")
plt.colorbar(im2, ax=axes[1, 1], label="Temperature (°C)")

# Time series at a point
point_lat, point_lon = 45.0, 0.0  # 45°N, 0°E (Europe)

# Find nearest grid points
orig_lat_idx = np.argmin(np.abs(source_lats - point_lat))
orig_lon_idx = np.argmin(np.abs(source_lons - point_lon))
regrid_lat_idx = np.argmin(np.abs(target_lats - point_lat))
regrid_lon_idx = np.argmin(np.abs(target_lons - point_lon))

orig_ts = dataset.temperature[:, orig_lat_idx, orig_lon_idx]
regrid_ts = full_result[:, regrid_lat_idx, regrid_lon_idx]

axes[1, 2].plot(orig_ts.time, orig_ts, "o-", label="Original 0.25°", alpha=0.7)
axes[1, 2].plot(regrid_ts.time, regrid_ts, "s-", label="Regridded 1°", alpha=0.7)
axes[1, 2].set_title(f"Time Series at {point_lat}°N, {point_lon}°E")
axes[1, 2].set_xlabel("Time")
axes[1, 2].set_ylabel("Temperature (°C)")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("weight_reuse_performance.png", dpi=150, bbox_inches="tight")
print("\nSaved plot as 'weight_reuse_performance.png'")

# Best practices summary
print("\n" + "=" * 60)
print("BEST PRACTICES SUMMARY")
print("=" * 60)

print("""
1. ALWAYS use weight reuse for repeated regridding:
   - Set reuse_weights=True
   - Specify a meaningful filename
   - Store weights on fast storage (SSD)

2. Workflow optimization:
   - Create regridder once, apply to multiple datasets
   - Process full time series together (not timestep by timestep)
   - Use meaningful weight filenames for different grid combinations

3. Performance expectations:
   - Weight generation: One-time cost (10-60s for large grids)
   - Weight loading: Very fast (0.1-2s)
   - Regridding application: Vectorized, ~100-1000x faster than loops

4. Storage considerations:
   - Weight files: ~10-100 MB for typical climate grids
   - Store weights with descriptive names
   - Clean up unused weight files periodically

5. Memory optimization:
   - Process in chunks if memory is limited
   - Use Dask for large datasets
   - Monitor memory usage during processing
""")

# Clean up
os.remove(weight_file)
print(f"\nCleaned up temporary weight file: {weight_file}")

# Show plot if running interactively
try:
    if __name__ == "__main__":
        plt.show()
except Exception:
    pass
