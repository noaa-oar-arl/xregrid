# NOAA RDHPCS SLURM Templates

This directory contains SLURM batch script templates for various NOAA RDHPCS systems.

## Templates

- `hera.sl`: Template for the Hera system.
- `jet.sl`: Template for the Jet system.
- `gaea_c5.sl`: Template for the Gaea C5 cluster.
- `gaea_c6.sl`: Template for the Gaea C6 cluster.

## Usage

1. Copy the appropriate template to your working directory.
2. Edit the script to specify your project account (`-A`), job name (`-J`), and other resource requirements.
3. Update the module loading section as needed for your application.
4. Submit the job using `sbatch`:

```bash
sbatch hera.sl
```

## System Specifics

### Hera
- **Default Partition**: `hera`
- **Cores per Node**: 40
- **Launcher**: `srun`

### Jet
- **Partitions**: `sjet`, `vjet`, `xjet`, `kjet`, `bigmem`, `service`
- **Recommended**: Do not specify a partition to allow random assignment from general compute resources, or specify one explicitly.
- **Cores per Node**: Varies by partition (e.g., `kjet` has 40).
- **Launcher**: `mpiexec`

### Gaea
- **Clusters**: `c5`, `c6`
- **Note**: Must specify the cluster using `-M <cluster>`.
- **Cores per Node**: 128 (C5), 192 (C6).
- **Launcher**: `srun`

For more detailed information, please refer to the [NOAA RDHPCS Documentation](https://docs.rdhpcs.noaa.gov).
